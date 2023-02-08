import numpy as np
import torch
import torch.nn as nn

from databoost.utils.general import AttrDict
from databoost.utils.model_utils import MultivariateGaussian
from garage.torch.policies import TanhGaussianMLPPolicy


class BCPolicy(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 n_hidden_layers: int,
                 dropout_rate: float):
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
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"obs_dim: {obs_dim}")
        net_layers = nn.ModuleList()
        net_layers.append(nn.Linear(obs_dim, hidden_dim))
        net_layers.append(nn.LayerNorm(hidden_dim))
        net_layers.append(nn.LeakyReLU())
        net_layers.append(nn.Dropout(p=dropout_rate))
        for _ in range(n_hidden_layers):
            net_layers.append(nn.Linear(hidden_dim, hidden_dim))
            net_layers.append(nn.LayerNorm(hidden_dim))
            net_layers.append(nn.LeakyReLU())
            net_layers.append(nn.Dropout(p=dropout_rate))
        net_layers.append(nn.Linear(hidden_dim, action_dim * 2))
        self.net = nn.Sequential(*net_layers).float().to(self.device)

        # Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
        self.apply(_weights_init)

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

    def embed(self, obs):
        with torch.no_grad():
            net = self.net.eval()[:-1]
            embs = net(obs)
        self.net.train()
        return embs


class TanhGaussianBCPolicy(TanhGaussianMLPPolicy):
    def __init__(self, obs_dim, act_dim, hidden_sizes=None, *args, **kwargs):
        env_spec = AttrDict({
            "observation_space": AttrDict({
                "flat_dim": obs_dim
            }),
            "action_space": AttrDict({
                "flat_dim": act_dim
            })
        }),
        self.num_hidden_layers = len(hidden_sizes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(hidden_sizes=hidden_sizes, env_spec=env_spec, *args, **kwargs)

    def forward(self, obs):
        dist, _ = super().forward(obs)
        return dist

    def loss(self,
             pred_act_dist,
             action_batch):
        '''Compute the mean negative log likelihood loss of the batch of actions.
        Args:
            pred_act_dist [databoost.utils.model_utils.MultivariateGaussian]:
                multivariate gaussian distribution(s) of predicted action
            action_batch [torch.Tensor]: batch of ground truth actions
        Returns:
            loss [torch.float]: mean negative log likelihood of batch
        '''
        if self.act_range is not None:
            action_batch /= self.act_range
        log_prob = pred_act_dist.log_prob(action_batch)
        loss = -1 * log_prob.mean()
        return loss

    def get_action(self, ob):
        '''Get action from the policy given an observation.
        Args:
            ob [np.ndarray]: an observation from the environment
        Returns:
            act [np.ndarray]: an action from the policy
        '''
        with torch.no_grad():
            ob = torch.tensor(ob[None], dtype=torch.float).to(self.device)
            dist = self._module(ob)
            act = dist.mean.cpu().detach().numpy()[0]
            if self.act_range is not None:
                act *= self.act_range
            return act

    def embed(self, obs):
        with torch.no_grad():
            x = obs
            for i, layer in enumerate(self._module._shared_mean_log_std_network._layers):
                if i + 1 == self.num_hidden_layers:
                    layer = layer[:-1]
                x = layer(x)
        return x
