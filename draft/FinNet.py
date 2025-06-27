import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io
from scipy.interpolate import griddata

# 检查GPU可用性并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 问题参数
L = 2.0
T = 1.0
Nt = 100
dt = T / Nt
nu = 0.01 / np.pi


# 修正后的时空编码
class SpatioTemporalEncoding(nn.Module):
    def __init__(self, d_model=256, max_time_steps=100):
        super().__init__()
        # 空间编码
        self.x_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, d_model)
        )
        # 时间编码
        self.t_embed = nn.Embedding(max_time_steps, d_model)

    def forward(self, x, t_steps):
        # x: (batch_size, 1), t_steps: (batch_size, 1)
        x_enc = self.x_encoder(x)  # (batch_size, d_model)
        t_enc = self.t_embed(t_steps.squeeze().long())  # (batch_size, d_model)
        return (x_enc + t_enc).unsqueeze(1)  # 添加序列维度 -> (batch_size, 1, d_model)


# 简化版Transformer架构（避免使用复杂的自注意力）
class BurgersTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = SpatioTemporalEncoding(d_model)

        # 使用简单的前馈网络代替复杂的Transformer层
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, t_steps):
        # x: (batch_size, 1), t_steps: (batch_size, 1)
        # 时空编码 (已包含序列维度)
        embedded = self.embedding(x, t_steps)  # (batch_size, 1, d_model)

        # 简单的前馈网络代替Transformer
        x = self.fc1(embedded.squeeze(1))  # (batch_size, d_model * 2)
        x = torch.relu(x)
        x = self.fc2(x)  # (batch_size, d_model)

        # 解码
        return self.decoder(x)  # (batch_size, 1)


# 初始化模型并移至设备
model = BurgersTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-4,
    step_size_up=2000,
    cycle_momentum=False
)

# 训练参数
num_points = 4096
epochs = 20000
losses = []
solutions = []

# 训练循环（GPU优化）
for epoch in range(epochs):
    optimizer.zero_grad()

    # 数据生成
    x = torch.rand(num_points, 1, device=device) * 2 - 1
    t_bias = torch.rand(num_points, device=device) ** 0.7
    t_steps = (t_bias * Nt).long().clamp(0, Nt - 1).view(-1, 1)

    # 初始条件约束
    mask_init = (t_steps == 0).squeeze()
    loss_init = torch.tensor(0.0, device=device)
    if mask_init.any():
        u_init = model(x[mask_init], t_steps[mask_init])
        loss_init = torch.mean((u_init + torch.sin(np.pi * x[mask_init])) ** 2)

    # 边界条件约束
    mask_bc = (x.abs() > 0.99).squeeze()
    loss_bc = torch.tensor(0.0, device=device)
    if mask_bc.any():
        u_bc = model(x[mask_bc], t_steps[mask_bc])
        loss_bc = torch.mean(u_bc ** 2)

    # PDE残差约束
    mask_pde = (~mask_init) & (~mask_bc)
    loss_pde = torch.tensor(0.0, device=device)
    if mask_pde.any():
        x_pde = x[mask_pde].clone().requires_grad_(True)
        t_pde = t_steps[mask_pde]

        u_pred = model(x_pde, t_pde)

        # 自动微分
        du_dx = torch.autograd.grad(u_pred, x_pde,
                                    grad_outputs=torch.ones_like(u_pred),
                                    create_graph=True)[0]


        def compute_second_order_derivative(du_dx, x_pde):
            # 使用后向差分计算二阶导数
            d2u_dx2 = torch.zeros_like(du_dx)
            d2u_dx2[1:-1] = (du_dx[2:] - 2 * du_dx[1:-1] + du_dx[:-2]) / (x_pde[2:] - x_pde[:-2]) ** 2
            # 对边界点使用单侧差分
            d2u_dx2[0] = (du_dx[1] - du_dx[0]) / (x_pde[1] - x_pde[0])  # 前向差分
            d2u_dx2[-1] = (du_dx[-1] - du_dx[-2]) / (x_pde[-1] - x_pde[-2])  # 后向差分
            return d2u_dx2

        # 使用有限差分计算二阶导数，处理边界点
        d2u_dx2 = compute_second_order_derivative(du_dx, x_pde)

        # 时间导数
        t_prev = (t_pde.float() - 1).clamp(min=0).long()
        u_prev = model(x_pde, t_prev)
        du_dt = (u_pred - u_prev) / dt

        residual = du_dt + u_pred * du_dx - nu * d2u_dx2
        loss_pde = torch.mean(residual ** 2)

    # 总损失
    total_loss = 2.0 * loss_init + 1.0 * loss_bc + loss_pde
    total_loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    # 记录损失
    losses.append(total_loss.item())

    # 保存结果
    if epoch % 100 == 0:
        print(f"Epoch {epoch:05d} | Loss: {total_loss.item():.3e}")

        with torch.no_grad():
            x_plot = torch.linspace(-1, 1, 200, device=device).view(-1, 1)
            t_plot = torch.full((200, 1), Nt // 2, device=device, dtype=torch.long)
            u_pred = model(x_plot, t_plot)
            solutions.append(u_pred.squeeze().cpu().numpy())


# 可视化训练损失
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# 可视化解的动画
fig, ax = plt.subplots()
x_plot = np.linspace(-1, 1, 100)
line, = ax.plot(x_plot, solutions[0])
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Burgers Equation Solution (Continuous Space)")

def update(frame):
    line.set_ydata(solutions[frame])
    return line,

ani = FuncAnimation(fig, update, frames=len(solutions), interval=200)

plt.subplot(1, 2, 2)
plt.show()


def plot_comparison(model, x_plot, t_steps):
    model.eval()
    data = scipy.io.loadmat('./Data/burgers_shock.mat')
    u_true = np.real(data['usol']).T
    x_true = data['x'].flatten()
    t_true = data['t'].flatten()

    X_pred, T_pred = np.meshgrid(x_plot.cpu().numpy().flatten(), np.linspace(0, 1, Nt))
    x_test = torch.tensor(X_pred.flatten(), dtype=torch.float32, device=device)
    t_test = torch.tensor(T_pred.flatten(), dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model(x_test.unsqueeze(1), t_test.unsqueeze(1))
        u_pred = u_pred.cpu().numpy().reshape(Nt, -1)  # 移回CPU

    # 插值到真实解网格
    points_pred = np.vstack([x_test, t_test]).T
    u_pred_interp = griddata(points_pred, u_pred.flatten(),
                             (np.meshgrid(x_true, t_true)),
                             method='cubic')

    # 计算误差
    error = np.abs(u_true - u_pred_interp)
    rel_error = np.linalg.norm(error) / np.linalg.norm(u_true)

    # 可视化对比
    fig = plt.figure(figsize=(18, 6))

    # 预测解
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(*np.meshgrid(x_true, t_true), u_pred_interp, cmap='viridis')
    ax1.set_title(f'Predicted Solution\nRel L2 Error: {rel_error:.2e}')

    # 真实解
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(*np.meshgrid(x_true, t_true), u_true, cmap='viridis')
    ax2.set_title('Ground Truth')

    # 误差分布
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(*np.meshgrid(x_true, t_true), error, cmap='hot')
    ax3.set_title('Absolute Error')

    plt.tight_layout()
    plt.show()



# 在训练结束后调用对比函数
print("\n正在进行解的质量验证...")
x_plot = torch.linspace(-1, 1, 200, device=device).view(-1,1)
t_plot = torch.arange(Nt, device=device).view(-1,1)
plot_comparison(model, x_plot, t_plot)
