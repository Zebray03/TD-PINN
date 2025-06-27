import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn

# 定义问题参数
L = 2.0  # 空间域 [-1, 1]
T = 1.0  # 总时间
Nt = 100  # 时间步数
dt = T / Nt  # 时间步长
nu = 0.01 / np.pi  # 粘性系数


# 定义神经网络
class BurgersContinuousNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.dt = dt
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x, t_step):
        t_norm = t_step.float() / Nt
        t_norm = t_norm.view(-1, 1)
        # 确保 x 的维度与 t_norm 一致
        if x.dim() != t_norm.dim():
            x = x.view(-1, 1)
        inputs = torch.cat([x, t_norm], dim=-1)
        return self.net(inputs)


# 初始化网络和优化器
model = BurgersContinuousNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

import time

# 任务1：不同num_points训练并记录时间和L2误差
unit = 1000
num_points_values = [1 * unit,
                     2 * unit,
                     3 * unit,
                     4 * unit,
                     5 * unit,
                     6 * unit,
                     7 * unit,
                     8 * unit,
                     9 * unit,
                     10 * unit]
epochs = 100000
training_results = []

for num_points in num_points_values:
    print(f"\n=== Training num_points={num_points} ===")

    # 重新初始化模型和优化器
    model = BurgersContinuousNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 随机采样
        x = torch.rand(num_points, 1) * 2 - 1
        t_steps = torch.randint(0, Nt, (num_points, 1))

        # 初始条件损失
        mask_init = (t_steps == 0)
        if mask_init.any():
            u_pred_init = model(x[mask_init.squeeze()], t_steps[mask_init.squeeze()])
            loss_init = torch.mean((u_pred_init + torch.sin(np.pi * x[mask_init.squeeze()])) ** 2)
        else:
            loss_init = torch.tensor(0.0)

        # 边界条件损失
        epsilon = 1e-2
        mask_bc = (x.abs() > 1 - epsilon)
        if mask_bc.any():
            u_pred_bc = model(x[mask_bc.squeeze()], t_steps[mask_bc.squeeze()])
            loss_bc = torch.mean(u_pred_bc ** 2)
        else:
            loss_bc = torch.tensor(0.0)

        # PDE残差损失
        mask_pde = (~mask_init) & (~mask_bc)
        if mask_pde.any():
            x_pde = x[mask_pde.squeeze()].requires_grad_(True)
            t_pde = t_steps[mask_pde.squeeze()]

            u_pred = model(x_pde, t_pde)
            du_dx = torch.autograd.grad(u_pred, x_pde, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
            d2u_dx2 = torch.autograd.grad(du_dx, x_pde, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
            u_prev = model(x_pde, t_pde - 1)
            du_dt = (u_pred - u_prev) / dt
            pde_residual = du_dt + u_pred * du_dx - nu * d2u_dx2
            loss_pde = torch.mean(pde_residual ** 2)
        else:
            loss_pde = torch.tensor(0.0)

        # 总损失
        total_loss = loss_init + loss_bc + 1e1 * loss_pde

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 打印训练进度
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

    training_time = time.time() - start_time

    # 计算预测解
    with torch.no_grad():
        x_plot = torch.linspace(-1, 1, 256).view(-1, 1)
        t_plot = torch.linspace(0, Nt, 100).long()
        X, T_mesh = np.meshgrid(x_plot.numpy(), t_plot.numpy())
        U = np.zeros_like(X)
        for i in range(len(t_plot)):
            u = model(x_plot, t_plot[i] * torch.ones_like(x_plot))
            U[i, :] = u.numpy().flatten()

    # 加载真实解
    try:
        data = scipy.io.loadmat('./Data/burgers_shock.mat')
        u_true = np.real(data['usol']).T
    except:
        u_true = -np.sin(np.pi * X) * np.exp(-T_mesh)

    # 计算L2相对误差
    error = np.abs(U - u_true)
    relative_error = np.linalg.norm(error) / np.linalg.norm(u_true)

    training_results.append((num_points, training_time, relative_error))
    print(f"num_points={num_points}, Time={training_time:.2f}s, L2 Error={relative_error:.4e}")

# 输出结果
print("\nResults Summary:")
print("num_points | Training Time (s) | Relative L2 Error")
for res in training_results:
    print(f"{res[0]:<9} | {res[1]:<16.2f} | {res[2]:.4e}")

# 保存结果到CSV

df = pd.DataFrame(training_results, columns=['num_points', 'training_time', 'relative_error'])
df.to_csv('training_results.csv', index=False)