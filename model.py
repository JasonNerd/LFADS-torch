import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, kl_divergence
from torch.distributions.transforms import AffineTransform
import abc
import math

def make_posterior(post_mean, post_std):
    return Independent(Normal(post_mean, post_std), 1)

def init_variance_scaling_(weight, scale_dim: int):
    scale_dim = torch.tensor(scale_dim)
    nn.init.normal_(weight, std=1 / torch.sqrt(scale_dim))

def init_linear_(linear: nn.Linear):
    init_variance_scaling_(linear.weight, linear.in_features)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)

def init_gru_cell_(cell: nn.GRUCell, scale_dim: int = None):
    if scale_dim is None:
        ih_scale = cell.input_size
        hh_scale = cell.hidden_size
    else:
        ih_scale = hh_scale = scale_dim
    init_variance_scaling_(cell.weight_ih, ih_scale)
    init_variance_scaling_(cell.weight_hh, hh_scale)
    nn.init.ones_(cell.bias_ih)
    cell.bias_ih.data[-cell.hidden_size :] = 0.0
    # NOTE: these weights are not present in TF
    nn.init.zeros_(cell.bias_hh)

class MultivariateNormal(nn.Module):
    def __init__(
        self,
        mean: float,
        variance: float,
        shape: int,
    ):
        super().__init__()
        # Create distribution parameter tensors
        means = torch.ones(shape) * mean
        logvars = torch.log(torch.ones(shape) * variance)
        self.mean = nn.Parameter(means, requires_grad=True)
        self.logvar = nn.Parameter(logvars, requires_grad=False)

    def make_posterior(self, post_mean, post_std):
        return Independent(Normal(post_mean, post_std), 1)

    def forward(self, post_mean, post_std):
        # Create the posterior distribution
        posterior = self.make_posterior(post_mean, post_std)
        # Create the prior and posterior
        prior_std = torch.exp(0.5 * self.logvar)
        prior = Independent(Normal(self.mean, prior_std), 1)
        # Compute KL analytically
        kl_batch = kl_divergence(posterior, prior)
        return torch.mean(kl_batch)


class AutoregressiveMultivariateNormal(nn.Module):
    def __init__(
        self,
        tau: float,
        nvar: float,
        shape: int,
    ):
        super().__init__()
        # Create the distribution parameters
        logtaus = torch.log(torch.ones(shape) * tau)
        lognvars = torch.log(torch.ones(shape) * nvar)
        self.logtaus = nn.Parameter(logtaus, requires_grad=True)
        self.lognvars = nn.Parameter(lognvars, requires_grad=True)

    def make_posterior(self, post_mean, post_std):
        return Independent(Normal(post_mean, post_std), 2)

    def log_prob(self, sample):
        # Compute alpha and process variance
        alphas = torch.exp(-1.0 / torch.exp(self.logtaus))
        logpvars = self.lognvars - torch.log(1 - alphas**2)
        # Create autocorrelative transformation
        transform = AffineTransform(loc=0, scale=alphas)
        # Align previous samples and compute means and stddevs
        prev_samp = torch.roll(sample, shifts=1, dims=1)
        means = transform(prev_samp)
        stddevs = torch.ones_like(means) * torch.exp(0.5 * self.lognvars)
        # Correct the first time point
        means[:, 0] = 0.0
        stddevs[:, 0] = torch.exp(0.5 * logpvars)
        # Create the prior and compute the log-probability
        prior = Independent(Normal(means, stddevs), 2)
        return prior.log_prob(sample)

    def forward(self, post_mean, post_std):
        posterior = self.make_posterior(post_mean, post_std)
        sample = posterior.rsample()
        log_q = posterior.log_prob(sample)
        log_p = self.log_prob(sample)
        kl_batch = log_q - log_p
        return torch.mean(kl_batch)


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)

class Reconstruction(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def reshape_output_params(self, output_params):
        pass

    @abc.abstractmethod
    def compute_loss(self, data, output_params):
        pass

    @abc.abstractmethod
    def compute_means(self, output_params):
        pass

class Poisson(nn.Module, Reconstruction):
    def __init__(self):
        super().__init__()
        self.n_params = 1

    def reshape_output_params(self, output_params):
        return torch.unsqueeze(output_params, dim=-1)

    def compute_loss(self, data, output_params):
        return F.poisson_nll_loss(
            output_params,
            data,
            full=True,
            reduction="none",
        )

    def compute_means(self, output_params):
        return torch.exp(output_params[..., 0])

class ClippedGRUCell(nn.GRUCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
        is_encoder: bool = False,
    ):
        super().__init__(input_size, hidden_size, bias=True)
        self.bias_hh.requires_grad = False
        self.clip_value = clip_value
        scale_dim = input_size + hidden_size if is_encoder else None
        init_gru_cell_(self, scale_dim=scale_dim)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        x_all = input @ self.weight_ih.T + self.bias_ih
        x_z, x_r, x_n = torch.chunk(x_all, chunks=3, dim=1)
        split_dims = [2 * self.hidden_size, self.hidden_size]
        weight_hh_zr, weight_hh_n = torch.split(self.weight_hh, split_dims)
        bias_hh_zr, bias_hh_n = torch.split(self.bias_hh, split_dims)
        h_all = hidden @ weight_hh_zr.T + bias_hh_zr
        h_z, h_r = torch.chunk(h_all, chunks=2, dim=1)
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h_n = (r * hidden) @ weight_hh_n.T + bias_hh_n
        n = torch.tanh(x_n + h_n)
        hidden = z * hidden + (1 - z) * n
        hidden = torch.clamp(hidden, -self.clip_value, self.clip_value)
        return hidden


class ClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.cell = ClippedGRUCell(
            input_size, hidden_size, clip_value=clip_value, is_encoder=True
        )

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden


class BidirectionalClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.fwd_gru = ClippedGRU(input_size, hidden_size, clip_value=clip_value)
        self.bwd_gru = ClippedGRU(input_size, hidden_size, clip_value=clip_value)

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        h0_fwd, h0_bwd = h_0
        input_fwd = input
        input_bwd = torch.flip(input, [1])
        output_fwd, hn_fwd = self.fwd_gru(input_fwd, h0_fwd)
        output_bwd, hn_bwd = self.bwd_gru(input_bwd, h0_bwd)
        output_bwd = torch.flip(output_bwd, [1])
        output = torch.cat([output_fwd, output_bwd], dim=2)
        h_n = torch.stack([hn_fwd, hn_bwd])
        return output, h_n

class Encoder(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hps = hparams
        # Initial hidden state for IC encoder
        self.ic_enc_h0 = nn.Parameter(
            torch.zeros((2, 1, hps.ic_enc_dim), requires_grad=True)
        )
        # Initial condition encoder
        self.ic_enc = BidirectionalClippedGRU(
            input_size=hps.encod_data_dim,
            hidden_size=hps.ic_enc_dim,
            clip_value=hps.cell_clip,
        )
        # Mapping from final IC encoder state to IC parameters
        self.ic_linear = nn.Linear(hps.ic_enc_dim * 2, hps.ic_dim * 2)
        init_linear_(self.ic_linear)
        # Decide whether to use the controller
        self.use_con = all(
            [
                hps.ci_enc_dim > 0,
                hps.con_dim > 0,
                hps.co_dim > 0,
            ]
        )
        if self.use_con:
            # Initial hidden state for CI encoder
            self.ci_enc_h0 = nn.Parameter(
                torch.zeros((2, 1, hps.ci_enc_dim), requires_grad=True)
            )
            # CI encoder
            self.ci_enc = BidirectionalClippedGRU(
                input_size=hps.encod_data_dim,
                hidden_size=hps.ci_enc_dim,
                clip_value=hps.cell_clip,
            )
        # Activation dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)
        print("---------- Encoder.__init__() ---------")

    def forward(self, data: torch.Tensor):
        hps = self.hparams
        batch_size = data.shape[0]
        assert data.shape[1] == hps.encod_seq_len, (
            f"Sequence length specified in HPs ({hps.encod_seq_len}) "
            f"must match data dim 1 ({data.shape[1]})."
        )
        data_drop = self.dropout(data)
        # option to use separate segment for IC encoding
        if hps.ic_enc_seq_len > 0:
            ic_enc_data = data_drop[:, : hps.ic_enc_seq_len, :]
            ci_enc_data = data_drop[:, hps.ic_enc_seq_len :, :]
        else:
            ic_enc_data = data_drop
            ci_enc_data = data_drop
        # Pass data through IC encoder
        ic_enc_h0 = torch.tile(self.ic_enc_h0, (1, batch_size, 1))
        _, h_n = self.ic_enc(ic_enc_data, ic_enc_h0)
        h_n = torch.cat([*h_n], dim=1)
        # Compute initial condition posterior
        h_n_drop = self.dropout(h_n)
        ic_params = self.ic_linear(h_n_drop)
        ic_mean, ic_logvar = torch.split(ic_params, hps.ic_dim, dim=1)
        ic_std = torch.sqrt(torch.exp(ic_logvar) + hps.ic_post_var_min)
        if self.use_con:
            # Pass data through CI encoder
            ci_enc_h0 = torch.tile(self.ci_enc_h0, (1, batch_size, 1))
            ci, _ = self.ci_enc(ci_enc_data, ci_enc_h0)
            # Add a lag to the controller input
            ci_fwd, ci_bwd = torch.split(ci, hps.ci_enc_dim, dim=2)
            ci_fwd = F.pad(ci_fwd, (0, 0, hps.ci_lag, 0, 0, 0))
            ci_bwd = F.pad(ci_bwd, (0, 0, 0, hps.ci_lag, 0, 0))
            ci_len = hps.encod_seq_len - hps.ic_enc_seq_len
            ci = torch.cat([ci_fwd[:, :ci_len, :], ci_bwd[:, -ci_len:, :]], dim=2)
            # Add extra zeros if necessary for forward prediction
            fwd_steps = hps.recon_seq_len - hps.encod_seq_len
            ci = F.pad(ci, (0, 0, 0, fwd_steps, 0, 0))
            # Add extra zeros if encoder does not see whole sequence
            ci = F.pad(ci, (0, 0, hps.ic_enc_seq_len, 0, 0, 0))
        else:
            # Create a placeholder if there's no controller
            ci = torch.zeros(data.shape[0], hps.recon_seq_len, 0).to(data.device)
        return ic_mean, ic_std, ci

class KernelNormalizedLinear(nn.Linear):
    def forward(self, input):
        normed_weight = F.normalize(self.weight, p=2, dim=1)
        return F.linear(input, normed_weight, self.bias)


class DecoderCell(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hps = hparams
        # Create the generator
        self.gen_cell = ClippedGRUCell(hps.co_dim, hps.gen_dim, clip_value=hps.cell_clip)
        # Create the mapping from generator states to factors
        self.fac_linear = KernelNormalizedLinear(hps.gen_dim, hps.fac_dim, bias=False)
        init_linear_(self.fac_linear)
        # Create the dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)
        # Decide whether to use the controller
        self.use_con = all([hps.ci_enc_dim > 0,hps.con_dim > 0,hps.co_dim > 0])
        if self.use_con:
            # Create the controller
            self.con_cell = ClippedGRUCell(
                2 * hps.ci_enc_dim + hps.fac_dim, hps.con_dim, clip_value=hps.cell_clip
            )
            # Define the mapping from controller state to controller output parameters
            self.co_linear = nn.Linear(hps.con_dim, hps.co_dim * 2)
            init_linear_(self.co_linear)
        # Keep track of the state dimensions
        self.state_dims = [
            hps.gen_dim,
            hps.con_dim,
            hps.co_dim,
            hps.co_dim,
            hps.co_dim,
            hps.fac_dim,
        ]

    def forward(self, input, h_0, sample_posteriors=True):
        hps = self.hparams

        # Split the state up into variables of interest
        gen_state, con_state, co_mean, co_std, gen_input, factor = torch.split(
            h_0, self.state_dims, dim=1
        )
        if self.use_con:
            # Compute controller inputs with dropout
            con_input = torch.cat([input, factor], dim=1)
            con_input_drop = self.dropout(con_input)
            # Compute and store the next hidden state of the controller
            con_state = self.con_cell(con_input_drop, con_state)
            # Compute the distribution of the controller outputs at this timestep
            co_params = self.co_linear(con_state)
            co_mean, co_logvar = torch.split(co_params, hps.co_dim, dim=1)
            co_std = torch.sqrt(torch.exp(co_logvar))
            # Sample from the distribution of controller outputs
            co_post = make_posterior(co_mean, co_std)
            gen_input = co_post.rsample() if sample_posteriors else co_mean
            
        # compute and store the next
        gen_state = self.gen_cell(gen_input, gen_state)
        gen_state_drop = self.dropout(gen_state)
        factor = self.fac_linear(gen_state_drop)

        hidden = torch.cat(
            [gen_state, con_state, co_mean, co_std, gen_input, factor], dim=1
        )

        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.cell = DecoderCell(hparams=hparams)

    def forward(self, input, h_0, sample_posteriors=True):
        hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden, sample_posteriors=sample_posteriors)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hps = hparams
        self.dropout = nn.Dropout(hps.dropout_rate)
        # Create the mapping from ICs to gen_state
        self.ic_to_g0 = nn.Linear(hps.ic_dim, hps.gen_dim)
        init_linear_(self.ic_to_g0)
        # Create the decoder RNN
        self.rnn = DecoderRNN(hparams=hparams)
        # Initial hidden state for controller
        self.con_h0 = nn.Parameter(torch.zeros((1, hps.con_dim), requires_grad=True))

    def forward(self, ic_samp, ci, sample_posteriors=True):
        """
            ic_samp.shape=torch.Size([256, 64])
            ci.shape=torch.Size([256, 45, 128])
            ext_input.shape=torch.Size([256, 35, 0])
        """
        hps = self.hparams
        batch_size = ic_samp.shape[0]
        # Calculate initial generator state and pass it to the RNN with dropout rate
        gen_init = self.ic_to_g0(ic_samp)
        gen_init_drop = self.dropout(gen_init)
        
        device = gen_init.device
        dec_rnn_h0 = torch.cat(
            [
                gen_init,
                torch.tile(self.con_h0, (batch_size, 1)),
                torch.zeros((batch_size, hps.co_dim), device=device),
                torch.ones((batch_size, hps.co_dim), device=device),
                torch.zeros((batch_size, hps.co_dim), device=device),
                self.rnn.cell.fac_linear(gen_init_drop),
            ],
            dim=1,
        )
        states, _ = self.rnn(
            ci, dec_rnn_h0, sample_posteriors=sample_posteriors
        )
        split_states = torch.split(states, self.rnn.cell.state_dims, dim=2)
        dec_output = (gen_init, *split_states)
        return dec_output

class LFADS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hparams = config
        self.recon = Poisson()
        self.use_con = all([config.ci_enc_dim > 0, config.con_dim > 0, config.co_dim > 0])
        self.ic_prior = MultivariateNormal(mean=0, variance=0.1, shape=config.ic_dim)
        self.co_prior = AutoregressiveMultivariateNormal(tau=10, nvar=0.1, shape=config.co_dim)
        self.readin = nn.Identity()
        self.encoder = Encoder(hparams=self.hparams)
        self.decoder = Decoder(hparams=self.hparams)
        self.readout = FanInLinear(config.fac_dim, config.recon_data_dim)

    def forward(self, batch: torch.Tensor, sample_posteriors: bool = False) -> torch.Tensor:
        """
        batch(b, t_in, n_in)[256, 35, 137]
        """
        encod_data = self.readin(batch)
        ic_mean, ic_std, ci = self.encoder(encod_data)
        ic_post = make_posterior(ic_mean, ic_std)
        ic_samp = ic_post.rsample() if sample_posteriors else ic_mean
        # Unroll the decoder to estimate latent states
        (
            gen_init,
            gen_states,
            con_states,
            co_means,
            co_stds,
            gen_inputs,
            factors,
        ) = self.decoder(ic_samp, ci, sample_posteriors=sample_posteriors)
        output_params = self.readout(factors)
        output_params = self.recon.reshape_output_params(output_params)
        output_params = self.recon.compute_means(output_params)
        return output_params, ic_mean, ic_std, co_means, co_stds

    def loss_func(self, outputs: torch.Tensor, batch: torch.Tensor):
        output_params, ic_mean, ic_std, co_means, co_stds = outputs
        l2 = compute_l2_penalty(self)
        recon = self.recon.compute_loss(batch, output_params)
        recon = torch.mean(recon)
        ic_kl = self.ic_prior(ic_mean, ic_std) * self.hparams.kl_ic_scale
        co_kl = self.co_prior(co_means, co_stds) * self.hparams.kl_co_scale
        # Compute ramping coefficients
        l2_ramp = compute_ramp(self.hparams.l2_start_epoch, self.hparams.l2_increase_epoch)
        kl_ramp = compute_ramp(self.hparams.kl_start_epoch, self.hparams.kl_increase_epoch)
        # Compute the final loss
        loss = self.hparams.loss_scale * (recon + l2_ramp * l2 + kl_ramp * (ic_kl + co_kl))
        return loss

def compute_l2_penalty(lfads):
    recurrent_kernels_and_weights = [
        (lfads.encoder.ic_enc.fwd_gru.cell.weight_hh, 0.0),
        (lfads.encoder.ic_enc.bwd_gru.cell.weight_hh, 0.0),
        (lfads.decoder.rnn.cell.gen_cell.weight_hh, 0.0),
    ]
    if lfads.use_con:
        recurrent_kernels_and_weights.extend(
            [
                (lfads.encoder.ci_enc.fwd_gru.cell.weight_hh, 0.0),
                (lfads.encoder.ci_enc.bwd_gru.cell.weight_hh, 0.0),
                (lfads.decoder.rnn.cell.con_cell.weight_hh, 0.0),
            ]
        )
    # Add recurrent penalty
    recurrent_penalty = 0.0
    recurrent_size = 0
    for kernel, weight in recurrent_kernels_and_weights:
        if weight > 0:
            recurrent_penalty += weight * 0.5 * torch.norm(kernel, 2) ** 2
            recurrent_size += kernel.numel()
    recurrent_penalty /= recurrent_size + 1e-8
    # Add recon penalty if applicable
    recon_penalty = 0.0
    if hasattr(lfads.recon, "compute_l2"):
        recon_penalty += lfads.recon.compute_l2()
    return recurrent_penalty + recon_penalty

def compute_ramp(start: int, increase: int, current_epoch=10):
        # Compute a coefficient that ramps from 0 to 1 over `increase` epochs
        ramp = (current_epoch + 1 - start) / (increase + 1)
        return torch.clamp(torch.tensor(ramp), 0, 1)

