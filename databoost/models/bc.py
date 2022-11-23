import torch
import torch.nn as nn

from databoost.utils.model_utils import MultivariateGaussian


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, n_hidden_layers):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net_layers = nn.ModuleList()
        net_layers.append(nn.Linear(obs_dim, hidden_dim))
        net_layers.append(nn.LayerNorm(hidden_dim))
        net_layers.append(nn.LeakyReLU())
        net_layers.append(nn.Dropout(p=0.2))
        for _ in range(n_hidden_layers):
            net_layers.append(nn.Linear(hidden_dim, hidden_dim))
            net_layers.append(nn.LayerNorm(hidden_dim))
            net_layers.append(nn.LeakyReLU())
            net_layers.append(nn.Dropout(p=0.2))
        net_layers.append(nn.Linear(hidden_dim, action_dim * 2))
        self.net = nn.Sequential(*net_layers).float().to(self.device)

    def forward(self, obs_batch):
        return MultivariateGaussian(self.net(obs_batch))