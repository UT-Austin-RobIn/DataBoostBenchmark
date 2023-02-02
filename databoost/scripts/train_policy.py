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

from databoost.base import DataBoostBenchmarkBase
from databoost.utils.general import AttrDict
from databoost.utils.data import dump_video_wandb


np.random.seed(42)
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
          n_steps: int,
          goal_condition: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.train().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4, betas=(0.9, 0.999))
    best_success_rate = 0

    model, _ = clip.load("ViT-B/32", device=device)

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
            obs_batch = obs_batch[:, 0, :].float()  # remove the window dimension, since just 1

            # append language instruction to observation
            #decoded_instructions = [bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")
            #                        for inst in traj_batch['infos']['instruction'][:, 0].data.cpu().numpy()]
            #text_tokens = clip.tokenize(decoded_instructions).to(device)#.float()  # [batch, 77]
            #text_tokens = model.encode_text(text_tokens)
            #obs_batch = torch.cat((obs_batch, text_tokens), dim=-1)

            pred_action_dist, _ = policy(obs_batch)
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
                # eval_loss = benchmark.validate(
                #     task_name=task_name,
                #     policy=policy,
                #     n_episodes=eval_episodes,
                #     goal_cond=goal_condition
                # )
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
                losses = []
                # if eval_loss <= best_eval_loss:
                if success_rate >= best_success_rate:
                    torch.save(copy.deepcopy(policy), os.path.join(dest_dir, f"{exp_name}-best.pt"))
                    best_success_rate = success_rate
            if step >= n_steps: break
    pbar.close()
    return policy

if __name__ == "__main__":
    import databoost
    from databoost.models.bc import BCPolicy, TanhGaussianBCPolicy

    benchmark_name = "language_table"
    task_name = "separate"
    boosting_method = "test7"
    goal_condition = False
    mask_goal_pos = False
    exp_name = f"{benchmark_name}-clipenc_dummy-{task_name}-{boosting_method}-goal_cond_{goal_condition}-mask_goal_pos_{mask_goal_pos}-4"
    dest_dir = f"/home/jullian-yapeter/data/DataBoostBenchmark/{benchmark_name}/models/dummy/{task_name}/{boosting_method}"

    benchmark_configs = {
        "benchmark_name": benchmark_name,
    }

    dataloader_configs = {
        #"dataset_dir": "/data/karl/data/table_sim/prior_data_clip",
        # "dataset_dir": "/home/karl/data/language_table/prior_data_clip",
        "dataset_dir": "/home/karl/data/language_table/seed_task_separate",
        "n_demos": None,
        "batch_size": 500,
        "seq_len": 1,
        "shuffle": True,
        "load_imgs": False,
        "goal_condition": goal_condition
    }

    # policy_configs = {
    #     #"obs_dim": 2048 + 77 * (2 if goal_condition else 1),
    #     "obs_dim": 2048 + 512,
    #     "action_dim": 2,
    #     "hidden_dim": 512,
    #     "n_hidden_layers": 4,
    #     "dropout_rate": 0.4
    # }
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

    train_configs = {
        "exp_name": exp_name,
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "dest_dir": dest_dir,
        "eval_period": 500,
        "eval_episodes": 1,
        "max_traj_len": 200,
        "n_steps": 1000,
        "goal_condition": goal_condition
    }

    eval_configs = {
        "task_name": task_name,
        "n_episodes": 1,
        "max_traj_len": 500,
        "goal_cond": goal_condition,
    }

    rollout_configs = {
        "task_name": task_name,
        "n_episodes": 3,
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
