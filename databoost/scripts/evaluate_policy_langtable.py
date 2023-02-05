import os
import random
import argparse

import torch
import numpy as np

import databoost
from databoost.utils.data import write_json
from databoost.models.iql.policies import GaussianPolicy, ConcatMlp
from torch import nn as nn
from databoost.models.iql.iql import IQLModel

random.seed(42)

policy_filename = "/data/sdass/DataBoostBenchmark/language_table/models/dummy/separate/IQL/language_table-separate-IQL-iql-testing"
benchmark = "language_table"
task = "separate"

obs_dim = 2048 + 512
action_dim = 2
qf_kwargs = dict(hidden_sizes=[512, 512], layer_norm=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_configs = {
    "policy": GaussianPolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_sizes=[512, 512, 512],
                    max_log_std=0,
                    min_log_std=-6,
                    std_architecture="values",
                    hidden_activation=nn.LeakyReLU(),
                    # output_activation=nn.Identity(),
                    layer_norm=True,
                ).to(device),
    "qf1": ConcatMlp(
                    input_size=obs_dim + action_dim,
                    output_size=1,
                    **qf_kwargs,
                ).to(device),
    "qf2": ConcatMlp(
                    input_size=obs_dim + action_dim,
                    output_size=1,
                    **qf_kwargs,
                ).to(device),
    "target_qf1": ConcatMlp(
                    input_size=obs_dim + action_dim,
                    output_size=1,
                    **qf_kwargs,
                ).to(device),
    "target_qf2": ConcatMlp(
                    input_size=obs_dim + action_dim,
                    output_size=1,
                    **qf_kwargs,
                ).to(device),
    "vf": ConcatMlp(
                    input_size=obs_dim,
                    output_size=1,
                    **qf_kwargs,
                ).to(device),

    "discount": 0.995,
    "quantile": 0.7,
    "clip_score": 100,
    "soft_target_tau": 0.005,
    "reward_scale": 10,
    "beta": 10.0,
    "policy_lr": 1e-3,
    "qf_lr": 1e-3,
    # "policy_weight_decay": 0.01,
    # "q_weight_decay": 0.01,
    # "optimizer_class": torch.optim.AdamW,
    "device": device
}

benchmark = databoost.get_benchmark(benchmark)
success_rates = []

# policy = torch.load(policy_filename)
policy = IQLModel(**policy_configs)
for i in range(3):
    checkpoint = torch.load(policy_filename + f'-best{i}.pt')
    policy.load_from_checkpoint(checkpoint, load_optimizer=True)
    success_rate, gif = benchmark.evaluate(
        task_name=task,
        policy=policy,
        n_episodes=30,
        max_traj_len=80,
        render=False,
        goal_cond=False
    )
    print(f"policy success rate: {success_rate}")
    success_rates.append(success_rate)
    print(f"avg success: {np.mean(success_rates)}")
    # metrics = {
    #     "window": n_window,
    #     "period": n_period,
    #     "n_episodes": n_episodes,
    #     "max": np.max(success_rates),
    #     "min": np.min(success_rates),
    #     "mean": np.mean(success_rates)
    # }
    # write_json(metrics, os.path.join(policy_dir, f"metrics-chkpt_-window_{int(n_window)}-per_{int(n_period)}.json"))
