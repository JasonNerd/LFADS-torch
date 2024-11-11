from lfads import LFADS
import torch
import os

def main(spikes: torch.Tensor):
    from yacs.config import CfgNode as CN
    config = CN()
    num_trials, seq_len, num_neurons = spikes.shape
    config.encod_data_dim = num_neurons
    config.encod_seq_len = seq_len
    config.recon_data_dim = num_neurons
    config.recon_seq_len = seq_len

    config.ic_enc_seq_len = 0
    config.ic_enc_dim = 64
    config.ci_enc_dim = 64
    config.ci_lag = 1
    config.con_dim = 64
    config.co_dim = 4
    config.ic_dim = 64
    config.gen_dim = 200
    config.fac_dim = 100
    config.dropout_rate = 0.2
    config.cell_clip=5.0
    config.ic_post_var_min=0.0001
    config.loss_scale=1.0
    config.kl_ic_scale = 1e-7
    config.kl_co_scale = 1e-7
    config.l2_start_epoch = 0
    config.l2_increase_epoch = 80
    config.kl_start_epoch = 0
    config.kl_increase_epoch = 80

    print("----------- Model Configuration ----------")
    print(config)
    model = LFADS(config)
    print("----------- Model Structures ----------")
    print(model)
    return [model]

def test():
    spikes = torch.randn((64, 128, 134))  # [batch_size, seq_len, num_neurons]
    model = main(spikes)[0]
    outputs = model(spikes)
    loss = model.loss_func(outputs, spikes)
    print(loss)

test()
