# import argparse
import os
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


random.seed(42)


def train(policy: nn.Module,
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
          goal_condition: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.train().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4, betas=(0.9, 0.999))
    best_success_rate = 0

    for epoch in tqdm(range(int(n_epochs))):
        losses = []
        for batch_num, traj_batch in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            obs_batch = traj_batch["observations"].to(device)
            obs_batch = obs_batch[:, 0, :]  # remove the window dimension, since just 1
            pred_action_dist = policy(obs_batch.float())
            action_batch = traj_batch["actions"].to(device)
            action_batch = action_batch[:, 0, :]  # remove the window dimension, since just 1
            loss = policy.loss(pred_action_dist, action_batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: loss = {np.mean(losses)}")
        wandb.log({"epoch": epoch, "loss": np.mean(losses)})
        if epoch % eval_period == 0:
            print(f"evaluating epoch {epoch} with {eval_episodes} episodes")
            success_rate, _ = benchmark.evaluate(
                task_name=task_name,
                policy=policy,
                n_episodes=eval_episodes,
                max_traj_len=max_traj_len,
                goal_cond=goal_condition,
                render=False
            )
            wandb.log({"epoch": epoch, "success_rate": success_rate})
            print(f"epoch {epoch}: success_rate = {success_rate}")
            if success_rate >= best_success_rate:
                torch.save(policy, os.path.join(dest_dir, f"{exp_name}-best.pt"))
                best_success_rate = success_rate
    return policy

if __name__ == "__main__":
    import databoost
    from databoost.models.bc import BCPolicy


    # '''temp'''
    # parser = argparse.ArgumentParser(
    #     description='temp')
    # parser.add_argument("--boosting_method", help="boosting method")
    # args = parser.parse_args()
    # ''''''

    # benchmark_name = "metaworld"
    # task_name = "pick-place-wall"
    # boosting_method = "large_seed-prior_fail"
    # goal_condition = True
    # mask_goal_pos = True
    # exp_name = f"{benchmark_name}-{task_name}-{boosting_method}-goal_cond_{goal_condition}-mask_goal_pos_{mask_goal_pos}-4"
    # dest_dir = f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/models/{task_name}/{boosting_method}"
    #
    # benchmark_configs = {
    #     "benchmark_name": benchmark_name,
    #     "mask_goal_pos": mask_goal_pos
    # }
    #
    # dataloader_configs = {
    #     "dataset_dir": [
    #         f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/large_seed/{task_name}",
    #         f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/prior/fail",
    #     ],
    #     "n_demos": None,
    #     "batch_size": 128,
    #     "seq_len": 1,
    #     "shuffle": True,
    #     "load_imgs": False,
    #     "goal_condition": goal_condition
    # }
    #
    # policy_configs = {
    #     "obs_dim": 39 * (2 if goal_condition else 1),
    #     "action_dim": 4,
    #     "hidden_dim": 512,
    #     "n_hidden_layers": 4,
    #     "dropout_rate": 0.4
    # }

    benchmark_name = "language_table"
    task_name = "None"
    boosting_method = "None"
    goal_condition = False
    mask_goal_pos = False
    exp_name = f"{benchmark_name}-{task_name}-{boosting_method}-goal_cond_{goal_condition}-mask_goal_pos_{mask_goal_pos}-4"
    dest_dir = f"/data/karl/experiments/DataBoostBenchmark/{benchmark_name}/models/{task_name}/{boosting_method}"

    benchmark_configs = {
        "benchmark_name": benchmark_name,
    }

    dataloader_configs = {
        "dataset_dir": "/data/karl/data/table_sim/prior_data",
        "n_demos": None,
        "batch_size": 128,
        "seq_len": 1,
        "shuffle": True,
        "load_imgs": False,
        "goal_condition": goal_condition
    }

    policy_configs = {
        "obs_dim": 2048 * (2 if goal_condition else 1),
        "action_dim": 2,
        "hidden_dim": 512,
        "n_hidden_layers": 4,
        "dropout_rate": 0.4
    }

    train_configs = {
        "exp_name": exp_name,
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "dest_dir": dest_dir,
        "eval_period": 10,
        "eval_episodes": 20,
        "max_traj_len": 500,
        "n_epochs": 100,
        "goal_condition": goal_condition
    }

    eval_configs = {
        "task_name": task_name,
        "n_episodes": 200,
        "max_traj_len": 500,
        "goal_cond": goal_condition,
    }

    rollout_configs = {
        "task_name": task_name,
        "n_episodes": 10,
        "max_traj_len": 500,
        "goal_cond": goal_condition,
    }

    configs = {
        "benchmark_configs": benchmark_configs,
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
        dir="/tmp/wandb_karl",
        entity="clvr",
        notes="",
    )

    os.makedirs(dest_dir, exist_ok=True)
    benchmark = databoost.get_benchmark(**benchmark_configs)
    env = benchmark.get_env(task_name)
    dataloader = env._get_dataloader(**dataloader_configs)

    policy = BCPolicy(**policy_configs)
    policy = train(policy=policy,
                   dataloader=dataloader,
                   benchmark=benchmark,
                   **train_configs)

    torch.save(policy, os.path.join(dest_dir, f"{exp_name}-last.pt"))
    success_rate, _ = benchmark.evaluate(
        policy=policy,
        render=False,
        **eval_configs
    )
    print(f"final success_rate: {success_rate}")
    wandb.log({"final_success_rate": success_rate})

    best_policy = torch.load(os.path.join(dest_dir, f"{exp_name}-best.pt"))
    success_rate, _ = benchmark.evaluate(
        policy=best_policy,
        render=False,
        **eval_configs
    )
    print(f"best success_rate: {success_rate}")
    wandb.log({"best_success_rate": success_rate})

    '''generate sample policy rollouts'''
    success_rate, gifs = benchmark.evaluate(
        policy=best_policy,
        render=True,
        **rollout_configs
    )
    dump_video_wandb(gifs, "rollouts")
