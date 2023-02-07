import numpy as np
import torch 
import torch.nn as nn

from utils import build_network

class V_Model(nn.Module):
    def __init__(self, model_cfg, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        architecture_v_encoder = model_cfg['architecture']['v_model']['encoder']
        architecture_v_decoder = model_cfg['architecture']['v_model']['decoder']

        self.v_encoder_conv = build_network(architecture_v_encoder['conv'])
        self.v_encoder_mean = build_network(architecture_v_encoder['mean'])
        self.v_encoder_log_var = build_network(architecture_v_encoder['log_var'])
        self.v_decoder = build_network(architecture_v_decoder)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.device)
        x = self.v_encoder_conv(x)
        mean = self.v_encoder_mean(x)
        log_var = self.v_encoder_log_var(x)
        z = self.add_noise(mean, log_var)
        return z

    def encoding(self, x):
        x = x.to(self.device)
        x = self.v_encoder_conv(x)
        mean = self.v_encoder_mean(x)
        log_var = self.v_encoder_log_var(x)
        return mean, log_var

    def decoding(self, x):
        x = x.to(self.device)
        x = self.v_decoder(x)
        return x

    def add_noise(self, mean, log_var):
        noise = torch.normal(0, 1, size=log_var.shape, device=self.device)
        std = torch.exp(0.5*log_var)
        return mean + std * noise


class C_Model(nn.Module):
    def __init__(self, model_cfg, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.q_value = 0.0

    def get_action(self, x):
        x = self.forward(x)
        if x.dim() == 2:
            x = x.squeeze(0)
        self.q_value = x
        action = torch.argmax(x).item()
        return action