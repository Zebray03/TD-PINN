"""
将Raissi源代码的TensorFlow架构替换为Pytorch的版本
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.io
from pyDOE import lhs
import time
from torch.optim import LBFGS

np.random.seed(0x89757)
torch.manual_seed(0x89757)


class PhysicsInformedNN(nn.Module):
    def __init__(self, layers, lb, ub, nu):
        super(PhysicsInformedNN, self).__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)
        self.nu = nu

        self.model = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.model.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                self.model.append(nn.Tanh())

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        for layer in self.model:
            X = layer(X)
        return X

    def net_u(self, x, t):
        return self(x, t)

    def net_f(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t + u * u_x - self.nu * u_xx
        return f

    def loss_fn(self, x_u, t_u, u_true, x_f, t_f):
        u_pred = self.net_u(x_u, t_u)
        mse_u = torch.mean((u_pred - u_true) ** 2)

        f_pred = self.net_f(x_f, t_f)
        mse_f = torch.mean(f_pred ** 2)

        return mse_u + mse_f


def main_loop(N_u, N_f, num_layers, num_neurons):
    nu = 0.01 / np.pi
    layers = [2] + [num_neurons] * num_layers + [1]

    data = scipy.io.loadmat('./Data/burgers_shock.mat')
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    x_u = torch.tensor(X_u_train[:, 0:1], dtype=torch.float32).requires_grad_(True)
    t_u = torch.tensor(X_u_train[:, 1:2], dtype=torch.float32).requires_grad_(True)
    u_true = torch.tensor(u_train, dtype=torch.float32)

    x_f = torch.tensor(X_f_train[:, 0:1], dtype=torch.float32).requires_grad_(True)
    t_f = torch.tensor(X_f_train[:, 1:2], dtype=torch.float32).requires_grad_(True)

    model = PhysicsInformedNN(layers, lb, ub, nu)

    optimizer = LBFGS(model.parameters(),
                      max_iter=50000,
                      max_eval=50000,
                      history_size=50,
                      line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = model.loss_fn(x_u, t_u, u_true, x_f, t_f)
        loss.backward()
        return loss

    start_time = time.time()
    optimizer.step(closure)
    elapsed = time.time() - start_time
    print(f'Training time: {elapsed:.4f}s')

    with torch.no_grad():
        X_star_tensor = torch.tensor(X_star, dtype=torch.float32)
        x_star = X_star_tensor[:, 0:1]
        t_star = X_star_tensor[:, 1:2]
        u_pred = model.net_u(x_star, t_star).numpy()

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    return error_u


if __name__ == "__main__":
    N_u = [20, 40, 60, 80, 100, 200]
    N_f = [2000, 4000, 6000, 7000, 8000, 10000]

    num_layers = [2, 4, 6, 8]
    num_neurons = [10, 20, 40]

    error_table_1 = np.zeros((len(N_u), len(N_f)))
    error_table_2 = np.zeros((len(num_layers), len(num_neurons)))

    # table1
    for i in range(len(N_u)):
        for j in range(len(N_f)):
            print(f"Begin to train N_u = {N_u[i]}, N_f = {N_f[j]}, num_layers = 9, num_neurons = 20")
            error_table_1[i, j] = main_loop(N_u[i], N_f[j], 9, 20)

    # table2
    for i in range(len(num_layers)):
        for j in range(len(num_neurons)):
            error_table_2[i,j] = main_loop(N_u[-1], N_f[-1], num_layers[i], num_neurons[j])

    np.savetxt('./error_table_1_pinn.csv', error_table_1, fmt='%.2e')
    np.savetxt('./tables/error_table_2_pinn.csv', error_table_2, fmt='%.2e')
