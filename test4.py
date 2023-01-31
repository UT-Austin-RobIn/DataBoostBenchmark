import os
import numpy as np
import torch
import torch.nn as nn
from databoost.models.bc import TanhGaussianBCPolicy
from databoost.utils.general import AttrDict
from databoost.utils.data import find_h5, read_h5


model_path = "/data/jullian-yapeter/DataBoostBenchmark/metaworld/models/pick-place-wall/mtsac_2500K-demos_50K-1/metaworld-pick-place-wall-mtsac_2500K-demos_50K-1-goal_cond_True-mask_goal_pos_True-best.pt"
goal_condition = True

policy_configs = {
    "env_spec": AttrDict({
        "observation_space": AttrDict({
            "flat_dim": 39 * (2 if goal_condition else 1)
        }),
        "action_space": AttrDict({
            "flat_dim": 4
        })
    }),
    "hidden_sizes": [400, 400, 400],
    "hidden_nonlinearity": nn.ReLU,
    "output_nonlinearity": None,
    "min_std": np.exp(-20.),
    "max_std": np.exp(2.)
}

policy = TanhGaussianBCPolicy(**policy_configs)
saved_state_dict = torch.load(model_path).state_dict()
policy.load_state_dict(saved_state_dict)
policy.eval()

filenames = find_h5("/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/seed/pick-place-wall")
traj = read_h5(filenames[0])
obs = traj.observations
obs[:, -3:] = 0
obs = np.concatenate((obs, obs), axis=-1)
obs = torch.tensor(obs, dtype=torch.float)

embedding = policy.embed(obs)
print(embedding[0])