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
from databoost.utils.general import AttrDict


random.seed(83)
np.random.seed(83)


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
    # best_eval_loss = float('inf')
    best_success_rate = 0

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
            pred_action_dist, _ = policy(obs_batch.float())
            action_batch = traj_batch["actions"].to(device)
            action_batch = action_batch[:, 0, :]  # remove the window dimension, since just 1
            loss = policy.loss(pred_action_dist, action_batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)
            if (step % eval_period) == 0:
                torch.save(copy.deepcopy(policy), os.path.join(dest_dir, f"{exp_name}-{step}.pt"))  
                if eval_episodes <= 0:
                    wandb.log({"step": step, "epoch": epoch, "loss": np.mean(losses)})
                    print(f"step {step}, epoch {epoch}: loss = {np.mean(losses):.3f}")
                    # continue
                # eval_loss = benchmark.validate(
                #     task_name=task_name,
                #     policy=policy,
                #     n_episodes=eval_episodes,
                #     goal_cond=goal_condition
                # )
                if step in (25000, 50000, 100000, 150000, 200000):
                    print(f"evaluating step {step} with {eval_episodes} episodes")
                    success_rate, _ = benchmark.evaluate(
                        policy=policy,
                        render=False,
                        task_name=task_name,
                        n_episodes=eval_episodes,
                        max_traj_len=500,
                        goal_cond=goal_condition
                    )
                    wandb.log({"step": step, "epoch": epoch, "loss": np.mean(losses), "success_rate": success_rate})
                    print(f"step {step}, epoch {epoch}: loss = {np.mean(losses):.3f}, success_rate = {success_rate}")
                    if success_rate >= best_success_rate:
                        torch.save(copy.deepcopy(policy), os.path.join(dest_dir, f"{exp_name}-best.pt"))
                        best_success_rate = success_rate
                else:
                    wandb.log({"step": step, "epoch": epoch, "loss": np.mean(losses)})
                    print(f"step {step}, epoch {epoch}: loss = {np.mean(losses):.3f}")
                losses = []
            if step >= n_steps: break
    pbar.close()

    return policy

if __name__ == "__main__":
    import databoost
    from databoost.models.bc import BCPolicy, TanhGaussianBCPolicy

    benchmark_name = "metaworld"
    task_name = "pick-place-wall"
    boosting_method = "NoTarget-ObservationsMasked"
    goal_condition = True
    mask_goal_pos = True
    exp_name = f"{benchmark_name}-1-{task_name}-{boosting_method}-goal_cond_{goal_condition}-mask_goal_pos_{mask_goal_pos}"
    dest_dir = f"/home/jullian-yapeter/data/models/{benchmark_name}/{task_name}/{boosting_method}-1"

    benchmark_configs = {
        "benchmark_name": benchmark_name,
        "mask_goal_pos": mask_goal_pos
    }

    dataloader_configs = {
        "dataset_dir": [
            # "/home/karl/data/language_table/seed_task_separate",
            # "/home/karl/data/language_table/prior_data_clip",
            # "/data/karl/data/language_table/rl_episodes"
            f"/home/jullian-yapeter/data/boosted_data/{benchmark_name}/{task_name}/{boosting_method}/data",
            # "/home/jullian-yapeter/data/boosted_data/language_table/separate/Handcraft/data"
        ],
        "n_demos": None,
        "batch_size": 500,
        "seq_len": 1,
        "shuffle": True,
        "load_imgs": False,
        "goal_condition": goal_condition
    }

    # policy_configs = {
    #     "obs_dim": 39 * (2 if goal_condition else 1),
    #     "action_dim": 4,
    #     "hidden_dim": 512,
    #     "n_hidden_layers": 4,
    #     "dropout_rate": 0.4
    # }
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

    train_configs = {
        "exp_name": exp_name,
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "dest_dir": dest_dir,
        "eval_period": 1e3,
        "eval_episodes": 300,
        "max_traj_len": 500,
        "n_steps": 2e5,
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
        dir="/home/jullian-yapeter/tmp",
        entity="clvr",
        notes="",
    )

    os.makedirs(dest_dir, exist_ok=True)
    benchmark = databoost.get_benchmark(**benchmark_configs)
    env = benchmark.get_env(task_name)
    dataloader = env._get_dataloader(**dataloader_configs)

    # policy = BCPolicy(**policy_configs)
    policy = TanhGaussianBCPolicy(**policy_configs)
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


###OLD

# '''temp'''
    # import argparse
    # parser = argparse.ArgumentParser(
    #     description='temp')
    # parser.add_argument("--n_task_demos", help="num seed groups")
    # parser.add_argument("--n_success", help="num success prior groups")
    # parser.add_argument("--n_seed", help="num seed groups")
    # args = parser.parse_args()
    # very_helpful_tasks = [
    #     "push-back",
    #     "pick-place",
    #     "hand-insert"
    # ]
    # helpful_tasks = [
    #     "push-back",
    #     "pick-place",
    #     "hand-insert",
    #     "push",
    #     "sweep",
    #     "coffee-push",
    #     "soccer",
    #     "sweep-into",
    #     "bin-picking",
    #     "peg-insert-side"
    # ] # 10 of them
    # harmful_tasks = [
    #     "lever-pull",
    #     "plate-slide",
    #     "assembly",
    #     "pick-out-of-hole",
    #     "coffee-pull",
    #     "handle-pull-side",
    #     "button-press-wall",
    #     "reach-wall",
    #     "button-press",
    #     "button-press-topdown"
    # ] # 10 of them
    # from databoost.envs.metaworld.config import tasks
    # all_tasks = list(tasks.keys())
    ##### Prior experiments
    # num_demos_per_group = 2000
    # num_success_groups = int(args.n_success)
    # num_fail_groups = 10 - num_success_groups
    ### with seed groups
    # num_seeds_per_group = 2
    # num_seed_groups = int(args.n_seed)
    #####
    ##### Oracle boost experiments
    # num_demos_per_group = 40
    # num_success_groups = int(args.n_success)
    #####
    ##### Splitsville
    # num_task_demos_per_group = 5
    # num_demo_groups = int(args.n_task_demos) // num_task_demos_per_group
    # ''''''

    # boosting_method = f"small_seed_{num_seed_groups * num_seeds_per_group}_+_success_{int(num_success_groups * num_demos_per_group)}_+_fail_{int(num_fail_groups * num_demos_per_group)}_6"
    # boosting_method = f"success_{int(num_success_groups * num_demos_per_group)}_+_fail_{int(num_fail_groups * num_demos_per_group)}_tanhGauss_3"
    # boosting_method = f"oracle_boost_{int(num_success_groups * num_demos_per_group)}_tanhGauss_6"
    # boosting_method = f"oracle_boost_{int(num_success_groups * num_demos_per_group)}_mtsac_gc_{10*5*50}_2"
    # boosting_method = f"oracle_boost_{int(num_success_groups * num_demos_per_group)}_mtsac_gc_{0}_2"
    # n_demo_steps = ["500", "1K", "2K", "3K", "5K", "10K", "20K", "30K", "50K"]
    # boosting_method = f"mtsac_2500K-demos_{n_demo_steps[int(args.n_groups)-1]}-1"
    # boosting_method = f"split-steps-rl_1e6-demos_perc_60"
    # boosting_method = f"split-oracle-{args.n_task_demos}-1"

    ##### RL experiments
    #     f"/home/jullian-yapeter/data/metaworld/metaworld_rl_v3_h5/{e}" for e in [4,8,12,18]
    # ] + [
    ##### Prior small
    #     f"/home/jullian-yapeter/data/{benchmark_name}/grouped_prior_small/success/{group_num + 1}_of_40" \
    #     for group_num in range(17)
    # ] + [
    ##### Prior experiments
    #     f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/grouped_prior/fail/{group_num + 1}_of_10" \
    #     for group_num in range(num_fail_groups)
    # ] + [
    #     f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/grouped_prior/success/{group_num + 1}_of_10" \
    #     for group_num in range(num_success_groups)
    # ] + [
    ##### Oracle experiments
    #     f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/grouped_seed/{group_num + 1}_of_10" \
    #     for group_num in range(num_success_groups)
    # ] + [
    ##### grouped exp experiments
    #     f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/grouped_seed_exp/{task_name}/{group_num + 1}_of_9" \
    #     for group_num in range(int(args.n_groups))
    # ] + [
    ##### seed
    #    f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark_name}/data/seed/{task_name}"
    ##### grouped small seed
        # f"/home/jullian-yapeter/data/{benchmark_name}/grouped_seed_small/{group_num + 1}_of_40" \
        # for group_num in range(1)
    ##### boosted dataset
        # f"/home/jullian-yapeter/data/boosted_data/{benchmark_name}/{task_name}/{boosting_method}/data"


    