import matplotlib.pyplot as plt
import numpy as np
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
        inputs = torch.cat([x, t_norm], dim=-1)
        return self.net(inputs)


# 初始化网络和优化器
model = BurgersContinuousNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练参数
num_points = 2000
epochs = 100000

# 初始化损失记录
total_losses = []
loss_icbc_list = []  # 初始+边界条件损失
loss_pde_list = []  # PDE残差损失

# 训练循环
for epoch in range(epochs):
    optimizer.zero_grad()

    # 随机采样
    x = torch.rand(num_points, 1) * 2 - 1
    t_steps = torch.randint(0, Nt, (num_points, 1))

    # 初始条件
    mask_init = (t_steps == 0)
    u_pred_init = model(x[mask_init.squeeze()], t_steps[mask_init.squeeze()])
    loss_init = torch.mean((u_pred_init + torch.sin(np.pi * x[mask_init.squeeze()])) ** 2)

    # 边界条件
    epsilon = 1e-2
    mask_bc = (x.abs() > 1 - epsilon)
    u_pred_bc = model(x[mask_bc.squeeze()], t_steps[mask_bc.squeeze()])
    loss_bc = torch.mean(u_pred_bc ** 2)

    # PDE残差
    mask_pde = (~mask_init) & (~mask_bc)
    x_pde = x[mask_pde.squeeze()].requires_grad_(True)
    t_pde = t_steps[mask_pde.squeeze()]

    u_pred = model(x_pde, t_pde)
    du_dx = torch.autograd.grad(u_pred, x_pde, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x_pde, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    u_prev = model(x_pde, t_pde - 1)
    du_dt = (u_pred - u_prev) / dt
    pde_residual = du_dt + u_pred * du_dx - nu * d2u_dx2
    loss_pde = torch.mean(pde_residual ** 2)

    # 总损失
    lambda_f = 1e1
    total_loss = loss_init + loss_bc + lambda_f * loss_pde

    # 记录损失
    total_losses.append(total_loss.item())
    loss_icbc_list.append(loss_init.item() + loss_bc.item())
    loss_pde_list.append(loss_pde.item())

    # 反向传播
    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

# 训练损失曲线
plt.figure(figsize=(10, 6))
plt.semilogy(total_losses, label='Total Loss', alpha=0.8)
plt.semilogy(loss_icbc_list, label='IC + BC Loss', alpha=0.8)
plt.semilogy(loss_pde_list, label='PDE Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss Components')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# 3D可视化
with torch.no_grad():  # 禁用梯度计算
    x_plot = torch.linspace(-1, 1, 256).view(-1, 1)
    t_plot = torch.linspace(0, Nt, 100).long()
    X, T_mesh = np.meshgrid(x_plot.numpy(), t_plot.numpy())
    U = np.zeros_like(X)

    for i in range(len(t_plot)):
        u = model(x_plot, t_plot[i] * torch.ones_like(x_plot))
        U[i, :] = u.detach().numpy().flatten()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T_mesh, U, cmap='viridis')
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x,t)")
    ax.set_title("Burgers Equation Solution")
    plt.show()

# 特定时间点比较
try:
    data = scipy.io.loadmat('./Data/burgers_shock.mat')
    u_true = np.real(data['usol']).T
    x_true = data['x'].flatten()
    t_true = data['t'].flatten()
except:
    x_true = np.linspace(-1, 1, 256)
    t_true = np.linspace(0, 1, 100)
    u_true = -np.sin(np.pi * X) * np.exp(-T_mesh)

plt.figure(figsize=(15, 5))
target_times = [0.25, 0.5, 0.75]
for idx, t in enumerate(target_times):
    # 预测解
    t_step = int(t * (Nt - 1))
    with torch.no_grad():  # 禁用梯度计算
        u_pred = model(x_plot, torch.full_like(x_plot, t_step)).detach().numpy()  # <--- 这里修复

    # 真实解
    t_idx = np.argmin(np.abs(t_true - t))
    u_true_t = u_true[t_idx] if u_true.shape[0] == len(t_true) else u_true[:, t_idx]

    plt.subplot(1, 3, idx + 1)
    plt.plot(x_plot.numpy(), u_pred, 'b-', lw=2, label='Predicted')
    plt.plot(x_true, u_true_t, 'r--', lw=2, label='True')
    plt.title(f't = {t:.2f}')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# 误差计算与可视化
error = np.abs(u_true - U)
relative_error = np.linalg.norm(error) / np.linalg.norm(u_true)
print(f"Relative L2 Error: {relative_error:.4e}")

plt.figure(figsize=(18, 12))
# 预测解
ax1 = plt.subplot(221, projection='3d')
ax1.plot_surface(X, T_mesh, U, cmap='viridis')
ax1.set_title('Predicted Solution', fontsize=20)
# 真实解
ax2 = plt.subplot(222, projection='3d')
ax2.plot_surface(X, T_mesh, u_true, cmap='viridis')
ax2.set_title('True Solution', fontsize=20)
# 误差
ax3 = plt.subplot(223, projection='3d')
ax3.plot_surface(X, T_mesh, error, cmap='Reds')
ax3.set_title('Absolute Error', fontsize=20)
# 等高线
ax4 = plt.subplot(224)
ax4.contourf(X, T_mesh, error, 50, cmap='Reds')
ax4.set_title('Error Contour', fontsize=20)
plt.tight_layout()
plt.show()
