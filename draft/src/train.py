import torch

from data import generate_data
from loss import compute_loss


def train(model, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=config.lr,
        step_size_up=2000,
        cycle_momentum=False
    )

    losses = []
    for epoch in range(config.epochs):
        optimizer.zero_grad()

        x, t_steps = generate_data(config)

        total_loss = compute_loss(model, x, t_steps, config)
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(total_loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch:05d} | Loss: {total_loss.item():.3e}")

    return losses
