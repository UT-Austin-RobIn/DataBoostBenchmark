import torch
import numpy as np

import databoost
from databoost.models.iql.policies import GaussianPolicy, ConcatMlp
from torch import nn as nn
from databoost.models.iql.iql import IQLModel


exp_name = "iql-og-again"
boosting_method = "demo"
policy_filename = f"CurateDataset/language_table/models/dummy/separate/{boosting_method}/{exp_name}/language_table-separate-{boosting_method}-{exp_name}"

benchmark = "language_table"
task = "separate"

n_chkpt = int(52e3)
n_window = 5
n_period = 2e3
n_episodes = 20

obs_dim = 2048 + 512
action_dim = 2
qf_kwargs = dict(hidden_sizes=[512]*3, layer_norm=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_configs = {
    "policy": GaussianPolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_sizes=[512]*4,
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

policy = IQLModel(**policy_configs)
for idx in range(n_window):
    chkpt = int(n_chkpt - idx * n_period)
    filename = policy_filename + f'-step{chkpt}.pt'
    print(f"Evaluating file {filename}")
    
    checkpoint = torch.load(filename)
    policy.load_from_checkpoint(checkpoint, load_optimizer=True)
    success_rate, gif = benchmark.evaluate(
        task_name=task,
        policy=policy,
        n_episodes=n_episodes,
        max_traj_len=80,
        render=False,
        goal_cond=False
    )
    print(f"policy success rate: {success_rate}")
    success_rates.append(success_rate)
    print(f"avg success: {np.mean(success_rates)}")
