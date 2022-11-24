import numpy as np
import torch
import torch.nn as nn

from databoost.utils.model_utils import MultivariateGaussian


class BCPolicy(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 n_hidden_layers: int):
        '''Baseline BC policy.

        Args:
            obs_dim [int]: the dimension of the observation (input) vector
            action_dim [int]: the dimension of the action (output) vector
            hidden_dim [int]: the dimension of each hidden layer
            n_hidden_layers [int]: the number of hidden layers
        Attributes:
            net [nn.Module]: network that takes obs as input and outputs
                             a vector of size action * 2 (parameters of action
                             MultivariateGaussian distribution)
        '''
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

    def forward(self, obs_batch: torch.Tensor) -> MultivariateGaussian:
        '''Perform forward propagation.

        Args:
            obs_batch [torch.Tensor]: input tensor of shape (batch_size, *obs_dim)
        Returns:
            act_dist [databoost.utils.model_utils.MultivariateGaussian]:
                multivariate gaussian distribution(s) of predicted action
        '''
        act_dist = MultivariateGaussian(self.net(obs_batch))
        return act_dist

    def loss(self,
             pred_act_dist: MultivariateGaussian,
             action_batch: torch.Tensor) -> torch.float:
        '''Compute the mean negative log likelihood loss of the batch of actions.

        Args:
            pred_act_dist [databoost.utils.model_utils.MultivariateGaussian]:
                multivariate gaussian distribution(s) of predicted action
            action_batch [torch.Tensor]: batch of ground truth actions
        Returns:
            loss [torch.float]: mean negative log likelihood of batch
        '''
        loss = pred_act_dist.nll(action_batch).mean()
        return loss

    def get_action(self, ob: np.ndarray) -> np.ndarray:
        '''Get action from the policy given an observation.

        Args:
            ob [np.ndarray]: an observation from the environment
        Returns:
            act [np.ndarray]: an action from the policy
        '''
        with torch.no_grad():
            ob = torch.tensor(ob[None], dtype=torch.float).to(self.device)
            act_dist = MultivariateGaussian(self.net(ob))
            act = act_dist.mu.cpu().detach().numpy()[0]
            return act
