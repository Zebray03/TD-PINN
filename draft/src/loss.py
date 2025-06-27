import numpy as np
import torch


def compute_second_order_derivative(du_dx, x_pde):
    d2u_dx2 = torch.zeros_like(du_dx)
    d2u_dx2[1:-1] = (du_dx[2:] - 2 * du_dx[1:-1] + du_dx[:-2]) / (x_pde[2:] - x_pde[:-2]) ** 2
    d2u_dx2[0] = (du_dx[1] - du_dx[0]) / (x_pde[1] - x_pde[0])
    d2u_dx2[-1] = (du_dx[-1] - du_dx[-2]) / (x_pde[-1] - x_pde[-2])
    return d2u_dx2


def compute_loss(model, x, t_steps, config):
    mask_init = (t_steps == 0).squeeze()
    loss_init = torch.tensor(0.0, device=config.device)
    if mask_init.any():
        u_init = model(x[mask_init], t_steps[mask_init])
        loss_init = torch.mean((u_init + torch.sin(np.pi * x[mask_init])) ** 2)

    mask_bc = (x.abs() > 0.99).squeeze()
    loss_bc = torch.tensor(0.0, device=config.device)
    if mask_bc.any():
        u_bc = model(x[mask_bc], t_steps[mask_bc])
        loss_bc = torch.mean(u_bc ** 2)

    mask_pde = (~mask_init) & (~mask_bc)
    loss_pde = torch.tensor(0.0, device=config.device)
    if mask_pde.any():
        x_pde = x[mask_pde].clone().requires_grad_(True)
        t_pde = t_steps[mask_pde]

        u_pred = model(x_pde, t_pde)
        du_dx = torch.autograd.grad(u_pred, x_pde, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]

        d2u_dx2 = compute_second_order_derivative(du_dx, x_pde)

        t_prev = (t_pde.float() - 1).clamp(min=0).long()
        u_prev = model(x_pde, t_prev)
        du_dt = (u_pred - u_prev) / config.dt

        residual = du_dt + u_pred * du_dx - config.nu * d2u_dx2
        loss_pde = torch.mean(residual ** 2)

    total_loss = config.loss_init_weight * loss_init + config.loss_bc_weight * loss_bc + config.loss_pde_weight * loss_pde
    return total_loss
