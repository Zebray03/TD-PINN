import torch
from torch import nn


class PINN(nn.Module):
    def __init__(self, latent_dim, lb, ub, nu):
        super(PINN, self).__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)
        self.nu = nu

        self.pinn = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, X_grid):
        z = self.pinn(X_grid)
        return z

    def net_u(self, X_grid, encoder, decoder):
        z = encoder(X_grid)
        z = self.pinn(z)
        u_pred = decoder(z)
        return u_pred

    def net_f(self, X_grid, encoder, decoder):
        X_grid.requires_grad_(True)
        u = self.net_u(X_grid, encoder, decoder)

        u_t = torch.autograd.grad(u, X_grid, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0][:, 1:2, :, :]

        u_x = torch.autograd.grad(u, X_grid, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0][:, 0:1, :, :]
        u_xx = torch.autograd.grad(u_x, X_grid, grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0][:, 0:1, :, :]

        f = u_t + u * u_x - self.nu * u_xx
        return f

    def loss_fn(self, X_u, u_true, X_f, encoder, decoder):
        u_pred = self.net_u(X_u, encoder, decoder)
        mse_u = torch.mean((u_pred - u_true) ** 2)

        f_pred = self.net_f(X_f, encoder, decoder)
        mse_f = torch.mean(f_pred ** 2)

        return mse_u + mse_f