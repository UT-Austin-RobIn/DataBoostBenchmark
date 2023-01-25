import copy
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


random.seed(88)
np.random.seed(88)


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
          n_steps: int,
          goal_condition: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.train().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4, betas=(0.9, 0.999))
    best_eval_loss = float('inf')

    step = 0
    epoch = 0
    losses = []

    n_steps = int(n_steps)
    pbar = tqdm(total=n_steps)
    while(step < n_steps):
        epoch += 1
        for batch_num, traj_batch in enumerate(dataloader):
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
            step += 1
            pbar.update(1)
            if (step % eval_period) == 0:                
                print(f"evaluating step {step} with {eval_episodes} episodes")
                eval_loss = benchmark.validate(
                    task_name=task_name,
                    policy=policy,
                    n_episodes=eval_episodes,
                    goal_cond=goal_condition
                )
                wandb.log({"step": step, "epoch": epoch, "loss": np.mean(losses), "eval_loss": eval_loss})
                print(f"step {step}, epoch {epoch}: loss = {np.mean(losses)}, eval_loss = {eval_loss}")
                losses = []
                if eval_loss <= best_eval_loss:
                    torch.save(copy.deepcopy(policy), os.path.join(dest_dir, f"{exp_name}-best.pt"))
                    best_eval_loss = eval_loss
            if step >= n_steps: break
        epoch += 1
    pbar.close()

    return policy

if __name__ == "__main__":
    import databoost
    from databoost.models.bc import BCPolicy


    # '''temp'''
    import argparse
    parser = argparse.ArgumentParser(
        description='temp')
    parser.add_argument("--n_success", help="num success prior groups")
    # parser.add_argument("--n_seed", help="num seed groups")
    args = parser.parse_args()
    very_helpful_tasks = [
        "push-back",
        "pick-place",
        "hand-insert"
    ]
    helpful_tasks = [
        "push-back",
        "pick-place",
        "hand-insert",
        "push",
        "sweep",
        "coffee-push",
        "soccer",
        "sweep-into",
        "bin-picking",
        "peg-insert-side"
    ] # 10 of them
    harmful_tasks = [
        "lever-pull",
        "plate-slide",
        "assembly",
        "pick-out-of-hole",
        "coffee-pull",
        "handle-pull-side",
        "button-press-wall",
        "reach-wall",
        "button-press",
        "button-press-topdown"
    ] # 10 of them
    from databoost.envs.metaworld.config import tasks
    all_tasks = list(tasks.keys())
    ##### Prior experiments
    num_demos_per_group = 2000
    num_success_groups = int(args.n_success)
    num_fail_groups = 10 - num_success_groups
    # num_seeds_per_group = 2
    # num_seed_groups = int(args.n_seed)
    #####
    ##### Oracle boost experiments
    # num_demos_per_group = 40
    # num_success_groups = int(args.n_success)
    #####
    # ''''''

    benchmark_name = "metaworld"
    task_name = "pick-place-wall"
    # boosting_method = f"small_seed_{num_seed_groups * num_seeds_per_group}_+_success_{int(num_success_groups * num_demos_per_group)}_+_fail_{int(num_fail_groups * num_demos_per_group)}_6"
    boosting_method = f"seed_3_+_success_{int(num_success_groups * num_demos_per_group)}_+_fail_{int(num_fail_groups * num_demos_per_group)}_2"
    # boosting_method = f"seed_3_+_oracle_boost_{int(num_success_groups * num_demos_per_group)}_1"
    goal_condition = True
    mask_goal_pos = True
    exp_name = f"{benchmark_name}-{task_name}-{boosting_method}-goal_cond_{goal_condition}-mask_goal_pos_{mask_goal_pos}"
    dest_dir = f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/models/{task_name}/{boosting_method}"

    benchmark_configs = {
        "benchmark_name": benchmark_name,
        "mask_goal_pos": mask_goal_pos
    }

    dataloader_configs = {
        # "dataset_dir": [
        #     # f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/large_seed/{task_name}",
        #     # f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/large_prior/success/{task_name}",
        #     # f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/large_prior/fail/{task_name}",
        #     # f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/large_prior/fail",
        #     # f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/large_prior/success",
        #     f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/large_prior/success/{args.task_data}",
        # ],
        # "dataset_dir": [
        #     f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/large_prior/success/{task}" \
        #     for task in very_helpful_tasks
        # ] + [
        #     f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/large_prior/success/{task}" \
        #     for task in harmful_tasks
        # ] + [
        #     f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/seed/{task_name}"
        # ],
        "dataset_dir": [
        ##### Prior experiments
            f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/grouped_prior/fail/{group_num + 1}_of_10" \
            for group_num in range(num_fail_groups)
        ] + [
            f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/grouped_prior/success/{group_num + 1}_of_10" \
            for group_num in range(num_success_groups)
        ] + [
        #####
        ##### Oracle experiments
        #     f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/grouped_seed/{group_num + 1}_of_10" \
        #     for group_num in range(num_success_groups)
        # ] + [
        #####
        ##### seed
           f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/small_seed/{task_name}"
        ##### grouped small seed
            # f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/grouped_small_seed/{group_num + 1}_of_5" \
            # for group_num in range(num_seed_groups)
        ],
        "n_demos": None,
        "batch_size": 128,
        "seq_len": 1,
        "shuffle": True,
        "load_imgs": False,
        "goal_condition": goal_condition
    }

    policy_configs = {
        "obs_dim": 39 * (2 if goal_condition else 1),
        "action_dim": 4,
        "hidden_dim": 512,
        "n_hidden_layers": 4,
        "dropout_rate": 0.4
    }

    train_configs = {
        "exp_name": exp_name,
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "dest_dir": dest_dir,
        "eval_period": 1e3,
        "eval_episodes": 100,
        "max_traj_len": 500,
        "n_steps": 5e5,
        "goal_condition": goal_condition
    }

    eval_configs = {
        "task_name": task_name,
        "n_episodes": 300,
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
        dir="/tmp",
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

    final_policy = copy.deepcopy(policy)
    torch.save(final_policy, os.path.join(dest_dir, f"{exp_name}-last.pt"))
    success_rate, _ = benchmark.evaluate(
        policy=final_policy,
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
