import os
import random

import torch
import torch.nn as nn
import wandb

import databoost
from databoost.models.iql.iql import IQLModel
from databoost.models.iql.policies import GaussianPolicy, ConcatMlp

random.seed(42)



# train/load a policy
benchmark_name = "metaworld"
task_name = "pick-place-wall"
version = "success10_fail90"
policy_dir = f"/data/sdass/DataBoostBenchmark/{benchmark_name}/models/all/"
policy_filename = f"{benchmark_name}-{task_name}-{version}-iql-term0.01-seed2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
goal_condition = False
obs_dim = 39 * (2 if goal_condition else 1)
action_dim = 4
qf_kwargs = dict(hidden_sizes=[512, 512, ], layer_norm=True)
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
        # "policy_weight_decay": 0,
        # "q_weight_decay": 0,
        "device": device
    }
policy = IQLModel(**policy_configs)

wandb.init(
        resume=policy_filename,
        project="boost",
        # config=configs,
        dir="/home/sdass/tmp",
        entity="clvr",
        notes="",
    )


for i in range(3):
    checkpoint = torch.load(os.path.join(policy_dir, policy_filename + f'-best{i}.pt'))
    policy.load_from_checkpoint(checkpoint, load_optimizer=True)
    print(f"evaluating {policy_filename + f'-best{i}.pt'} on {task_name} task")
    
    # initialize appropriate benchmark with corresponding task
    benchmark = databoost.get_benchmark(benchmark_name)
    
    # evaluate the policy using the benchmark
    success_rate, gif = benchmark.evaluate(
        task_name=task_name,
        policy=policy,
        n_episodes=300,
        max_traj_len=500,
        render=False
    )
    print(f"policy success rate: {success_rate}")
    wandb.log({f"best_success_rate{i}": success_rate})


# checkpoint = torch.load(os.path.join(policy_dir, policy_filename + f'-last.pt'))
# policy.load_from_checkpoint(checkpoint, load_optimizer=True)
# print(f"evaluating {policy_filename + f'-last.pt'} on {task_name} task")

# # initialize appropriate benchmark with corresponding task
# benchmark = databoost.get_benchmark(benchmark_name)

# # evaluate the policy using the benchmark
# success_rate, gif = benchmark.evaluate(
#     task_name=task_name,
#     policy=policy,
#     n_episodes=300,
#     max_traj_len=500,
#     render=False
# )
# print(f"policy success rate: {success_rate}")
# wandb.log({f"last_success_rate": success_rate})