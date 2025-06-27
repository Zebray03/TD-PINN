import torch
import torch.nn as nn


class SpatioTemporalEncoding(nn.Module):
    def __init__(self, d_model=256, max_time_steps=100):
        super().__init__()
        self.x_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, d_model)
        )
        self.t_embed = nn.Embedding(max_time_steps, d_model)

    def forward(self, x, t_steps):
        x_enc = self.x_encoder(x)
        t_enc = self.t_embed(t_steps.squeeze().long())
        return (x_enc + t_enc).unsqueeze(1)



class BurgersTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = SpatioTemporalEncoding(config.d_model, config.max_time_steps)

        self.fc1 = nn.Linear(config.d_model, config.d_model * 2)
        self.fc2 = nn.Linear(config.d_model * 2, config.d_model)

        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1)
        )

    def forward(self, x, t_steps):
        embedded = self.embedding(x, t_steps)
        x = self.fc1(embedded.squeeze(1))
        x = torch.relu(x)
        x = self.fc2(x)
        return self.decoder(x)
