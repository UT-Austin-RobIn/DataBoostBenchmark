# import argparse
import clip
import os
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import databoost
from databoost.base import DataBoostBenchmarkBase
from databoost.utils.general import AttrDict
from databoost.utils.data import dump_video_wandb
from databoost.models.bc import BCPolicy, TanhGaussianBCPolicy


np.random.seed(23)

benchmark_name = "language_table"
task_name = "separate"
boosting_method = "Handcraft"
goal_condition = False
mask_goal_pos = False
exp_name = f"{benchmark_name}-{task_name}-{boosting_method}-goal_cond_{goal_condition}-mask_goal_pos_{mask_goal_pos}"
dest_dir = f"/home/jullian-yapeter/data/DataBoostBenchmark/{benchmark_name}/models/dummy/{task_name}/{boosting_method}"


benchmark_configs = {
    "benchmark_name": benchmark_name,
}

policy_configs = {
    "env_spec": AttrDict({
        "observation_space": AttrDict({
            "flat_dim": 2048 + 512
        }),
        "action_space": AttrDict({
            "flat_dim": 2
        })
    }),
    "hidden_sizes": [400, 400, 400],
    "hidden_nonlinearity": nn.ReLU,
    "output_nonlinearity": None,
    "min_std": np.exp(-20.),
    "max_std": np.exp(2.)
}

eval_configs = {
    "task_name": task_name,
    "n_episodes": 20,
    "max_traj_len": 100,
    "goal_cond": goal_condition,
}

benchmark = databoost.get_benchmark(**benchmark_configs)
env = benchmark.get_env(task_name)

for idx in []
policy = torch.load(os.path.join(dest_dir, f"{exp_name}-{200000}.pt"))
success_rate, _ = benchmark.evaluate(
    policy=policy,
    render=False,
    **eval_configs
)
print(f"success_rate: {success_rate}")