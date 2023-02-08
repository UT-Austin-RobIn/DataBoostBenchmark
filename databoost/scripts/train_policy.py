import os

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


def train(policy: nn.Module,
          dataloader: DataLoader,
          benchmark: DataBoostBenchmarkBase,
          exp_name: str,
          task_name: str,
          dest_dir: str,
          eval_period: int,
          eval_episodes: int,
          eval_max_traj_len: int,
          chkpt_save_period: int,
          n_steps: int,
          goal_condition: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.train().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4, betas=(0.9, 0.999))
    best_success_rate = 0

    step = 0
    epoch = 0
    losses = []

    n_steps = int(n_steps)
    pbar = tqdm(total=n_steps)
    while (step < n_steps):
        epoch += 1
        for _, traj_batch in enumerate(dataloader):
            optimizer.zero_grad()
            obs_batch = traj_batch["observations"].to(device)
            # remove the window dimension, since just 1
            obs_batch = obs_batch[:, 0, :]
            pred_action_dist = policy(obs_batch.float())
            action_batch = traj_batch["actions"].to(device)
            # remove the window dimension, since just 1
            action_batch = action_batch[:, 0, :]
            loss = policy.loss(pred_action_dist, action_batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)
            if (step % chkpt_save_period) == 0:
                torch.save(policy.detach().clone(), os.path.join(
                    dest_dir, f"{exp_name}-{step}.pt"))
            if (step % eval_period) == 0:
                if eval_episodes <= 0 or eval_max_traj_len <= 0:
                    wandb.log({"step": step, "epoch": epoch,
                              "loss": np.mean(losses)})
                    print(
                        f"step {step}, epoch {epoch}: loss = {np.mean(losses):.3f}")
                    continue
                print(f"evaluating step {step} with {eval_episodes} episodes")
                success_rate, _ = benchmark.evaluate(
                    policy=policy,
                    render=False,
                    task_name=task_name,
                    n_episodes=eval_episodes,
                    max_traj_len=eval_max_traj_len,
                    goal_cond=goal_condition
                )
                wandb.log({"step": step, "epoch": epoch, "loss": np.mean(
                    losses), "success_rate": success_rate})
                print(
                    f"step {step}, epoch {epoch}: loss = {np.mean(losses):.3f}, success_rate = {success_rate}")
                if success_rate >= best_success_rate:
                    torch.save(policy.detach().clone(), os.path.join(
                        dest_dir, f"{exp_name}-best.pt"))
                    best_success_rate = success_rate
                losses = []
            if step >= n_steps:
                break
    pbar.close()
    return policy


if __name__ == "__main__":
    import databoost
    from databoost.models.bc import TanhGaussianBCPolicy

    '''Set training configuration'''
    benchmark_name = "metaworld"
    task_name = "pick-place-wall"
    experiment_method = "R3M"
    policy_class = TanhGaussianBCPolicy
    goal_condition = True
    exp_name = f"{benchmark_name}-{task_name}-{experiment_method}"
    dest_dir = f"trained_models/{benchmark_name}/{task_name}/{experiment_method}"

    wandb_config = {
        "project": "boost",
        "dir": "/tmp",
        "entity": "sample_lab",
        "notes": "",
    }

    benchmark_configs = {
        "benchmark_name": benchmark_name,
    }

    dataloader_configs = {
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
        "obs_dim": 39 * (2 if goal_condition else 1),
        "act_dim": 4,
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
        "eval_episodes": 20,
        "eval_max_traj_len": 500,
        "chkpt_save_period": 5e3,
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
        "n_episodes": 10,  # 10,
        "max_traj_len": 120,  # 120,
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
        config=configs,
        **wandb_config
    )

    '''Build training setup and train'''
    os.makedirs(dest_dir, exist_ok=True)
    benchmark = databoost.get_benchmark(**benchmark_configs)
    env = benchmark.get_env(task_name)
    dataloader = env.get_combined_dataloader(**dataloader_configs)

    policy = policy_class(**policy_configs)
    policy = train(policy=policy,
                   dataloader=dataloader,
                   benchmark=benchmark,
                   **train_configs)

    '''Test the final checkpoint of the trained policy'''
    final_policy = policy.detach().clone()
    torch.save(final_policy, os.path.join(dest_dir, f"{exp_name}-last.pt"))
    success_rate, _ = benchmark.evaluate(
        policy=final_policy,
        render=False,
        **eval_configs
    )
    print(f"final success_rate: {success_rate}")
    wandb.log({"final_success_rate": success_rate})

    '''Test the best performing checkpoint of the trained policy'''
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
        policy=policy,
        render=True,
        **rollout_configs
    )
    dump_video_wandb(gifs, "rollouts")
