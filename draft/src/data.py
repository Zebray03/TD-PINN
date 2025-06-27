import torch


def generate_data(config):
    x = torch.rand(config.num_points, 1, device=config.device) * 2 - 1
    t_bias = torch.rand(config.num_points, device=config.device) ** 0.7
    t_steps = (t_bias * config.Nt).long().clamp(0, config.Nt - 1).view(-1, 1)
    return x, t_steps
