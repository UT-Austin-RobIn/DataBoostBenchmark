# import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from databoost.base import DataBoostBenchmarkBase
from databoost.utils.data import dump_video_wandb
from databoost.models.iql.policies import GaussianPolicy, ConcatMlp
from databoost.models.iql.iql import IQLTrainer

random.seed(42)

def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(trainer,
          dataloader: DataLoader,
          benchmark: DataBoostBenchmarkBase,
          exp_name: str,
          benchmark_name: str,
          task_name: str,
          dest_dir: str,
          eval_period: int,
          eval_episodes: int,
          max_traj_len: int,
          n_epochs: int,
          n_epoch_cycles: int,
          goal_condition: bool = False):

    best_success_rate = 0
    global_step = 0
    for epoch in tqdm(range(int(n_epochs))):
        for cycle in range(n_epoch_cycles):
            logs = {}
            for batch_num, batch in enumerate(dataloader):
                log = trainer.train_from_torch(batch)
                for k in log:
                    logs[k] = logs[k]+(log[k]/len(dataloader)) if k in logs else log[k]/len(dataloader)
                global_step += 1
            wandb.log(logs, step=global_step)

        if epoch % eval_period == 0:
            print(f"evaluating epoch {epoch} with {eval_episodes} episodes")
            success_rate, gifs = benchmark.evaluate(
                task_name=task_name,
                policy=trainer.policy,
                n_episodes=eval_episodes,
                max_traj_len=max_traj_len,
                goal_cond=goal_condition,
                render=False
            )

            wandb.log({"success_rate": success_rate}, step=global_step)
            print(f"epoch {epoch}: success_rate = {success_rate}")
            if success_rate >= best_success_rate:
                torch.save(trainer.policy, os.path.join(dest_dir, f"{exp_name}-best.pt"))
                best_success_rate = success_rate
    return trainer

if __name__ == "__main__":
    import databoost
    from databoost.models.bc import BCPolicy


    # '''temp'''
    # parser = argparse.ArgumentParser(
    #     description='temp')
    # parser.add_argument("--boosting_method", help="boosting method")
    # args = parser.parse_args()
    # ''''''

    benchmark_name = "metaworld"
    task_name = "pick-place-wall"
    boosting_method = ""
    version = "seed_and_prior_success"
    exp_name = f"{benchmark_name}-{task_name}-{version}-{sys.argv[1]}"
    # dest_dir = f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/models/{task_name}/{boosting_method}"
    dest_dir = f"/data/sdass/DataBoostBenchmark/{benchmark_name}/models/all"
    goal_condition = False

    dataloader_configs = {
        # "dataset_dir": f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/seed/{task_name}",
        # "dataset_dir": f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/boosted_data/{task_name}/{boosting_method}",
        "dataset_dir": f"/home/sdass/boosting/data/pick_place_wall/{version}",
        "n_demos": None,
        "batch_size": 256,
        "seq_len": 2,
        "shuffle": True,
        "goal_condition": goal_condition
    }

    # policy_configs = {
    #     "obs_dim": 39 * (2 if goal_condition else 1),
    #     "action_dim": 4,
    #     "hidden_dim": 512,
    #     "n_hidden_layers": 4,
    #     "dropout_rate": 0.4
    # }

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
        "quantile": 0.9,
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

    train_configs = {
        "exp_name": exp_name,
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "dest_dir": dest_dir,
        "eval_period": 5,
        "eval_episodes": 20,
        "max_traj_len": 500,
        "n_epochs": 1000, #iql : 1000
        "n_epoch_cycles": 5,
        "goal_condition": goal_condition
    }

    eval_configs = {
        "task_name": task_name,
        "n_episodes": 300,
        "max_traj_len": 500,
        "goal_cond": goal_condition
    }

    rollout_configs = {
        "task_name": task_name,
        "n_episodes": 10,
        "max_traj_len": 500,
        "goal_cond": goal_condition
    }

    configs = {
        "dataloader_configs": dataloader_configs,
        "policy_configs": policy_configs,
        "train_configs": train_configs,
        "eval_configs": eval_configs,
        "rollout_configs": rollout_configs
    }

    wandb.init(
        resume=exp_name,
        project="boost",
        config=configs,
        dir="/home/sdass/tmp",
        entity="clvr",
        notes="",
    )

    os.makedirs(dest_dir, exist_ok=True)
    benchmark = databoost.get_benchmark(benchmark_name)
    env = benchmark.get_env(task_name)
    dataloader = env._get_dataloader(**dataloader_configs)

    policy = train(trainer=IQLTrainer(**policy_configs),
                   dataloader=dataloader,
                   benchmark=benchmark,
                   **train_configs).policy

    torch.save(policy, os.path.join(dest_dir, f"{exp_name}-last.pt"))
    # success_rate, _ = benchmark.evaluate(
    #     policy=policy,
    #     render=False,
    #     **eval_configs
    # )
    # print(f"final success_rate: {success_rate}")
    # wandb.log({"final_success_rate": success_rate})

    best_policy = torch.load(os.path.join(dest_dir, f"{exp_name}-best.pt"))
    success_rate, _ = benchmark.evaluate(
        policy=best_policy,
        render=False,
        **eval_configs
    )
    print(f"best success_rate: {success_rate}")
    wandb.log({"best_success_rate": success_rate})

    # '''generate sample policy rollouts'''
    success_rate, gifs = benchmark.evaluate(
        policy=best_policy,
        render=True,
        **rollout_configs
    )
    dump_video_wandb(gifs, "rollouts")
