import numpy as np
import torch


class Config:
    def __init__(self):
        self.L = 2.0
        self.T = 1.0
        self.Nt = 100
        self.dt = self.T / self.Nt
        self.nu = 0.01 / np.pi
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = 256
        self.max_time_steps = 100
        self.num_points = 4096
        self.epochs = 20000
        self.lr = 1e-4
        self.weight_decay = 1e-6
        self.loss_init_weight = 2.0
        self.loss_bc_weight = 1.0
        self.loss_pde_weight = 1.0
