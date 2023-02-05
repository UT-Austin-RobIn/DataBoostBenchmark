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
                if step in (5000, 125000, 250000, 375000, 500000):
                    print(f"evaluating step {step} with {eval_episodes} episodes")
                    success_rate, _ = benchmark.evaluate(
                        policy=policy,
                        render=False,
                        task_name=task_name,
                        n_episodes=eval_episodes,
                        max_traj_len=120,
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

    benchmark_name = "language_table"
    task_name = "separate"
    boosting_method = "BC"
    goal_condition = False
    mask_goal_pos = False
    exp_name = f"{benchmark_name}-{task_name}-{boosting_method}-goal_cond_{goal_condition}-mask_goal_pos_{mask_goal_pos}"
    dest_dir = f"/home/jullian-yapeter/data/DataBoostBenchmark/{benchmark_name}/models/dummy/{task_name}/{boosting_method}"

    benchmark_configs = {
        "benchmark_name": benchmark_name,
    }

    dataloader_configs = {
        #"dataset_dir": "/data/karl/data/table_sim/prior_data_clip",
        # "dataset_dir": "/home/karl/data/language_table/prior_data_clip",
        # "dataset_dir": [
        #     "/home/karl/data/language_table/prior_data_clip",
        #     "/data/karl/data/language_table/rl_episodes"
        # ],
        # "/home/karl/data/language_table/seed_task_separate",
        # "/home/karl/data/language_table/prior_data_clip",
        # "/data/karl/data/language_table/rl_episodes"
        # f"/home/jullian-yapeter/data/boosted_data/{benchmark_name}/{task_name}/{boosting_method}/data",
        # "/home/jullian-yapeter/data/boosted_data/language_table/separate/Handcraft/data"
        "dataset_dir": [
            f"/home/jullian-yapeter/data/boosted_data/language_table/separate/{boosting_method}/part_0/data/seed",
        ] + [
            f"/home/jullian-yapeter/data/boosted_data/language_table/separate/{boosting_method}/part_{idx}/data/retrieved" \
            for idx in range(5)
        ],
        "n_demos": None,
        "batch_size": 128,
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
    # policy_configs = {
    #     "env_spec": AttrDict({
    #         "observation_space": AttrDict({
    #             "flat_dim": 2048 + 512
    #         }),
    #         "action_space": AttrDict({
    #             "flat_dim": 2
    #         })
    #     }),
    #     "hidden_sizes": [512, 512, 512, 512],
    #     "hidden_nonlinearity": nn.ReLU,
    #     "output_nonlinearity": None,
    #     "min_std": np.exp(-20.),
    #     "max_std": np.exp(2.)
    # }

    obs_dim = 2048 + 512
    action_dim = 2
    qf_kwargs = dict(hidden_sizes=[512, 512], layer_norm=True)

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

    train_configs = {
        "exp_name": exp_name,
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "dest_dir": dest_dir,
        "eval_period": 1e3,
        "eval_episodes": 10,
        "max_traj_len": 120,
        "n_steps": 5e5,
        "goal_condition": goal_condition
    }

    eval_configs = {
        "task_name": task_name,
        "n_episodes": 50,
        "max_traj_len": 120,
        "goal_cond": goal_condition,
    }

    rollout_configs = {
        "task_name": task_name,
        "n_episodes": 10,
        "max_traj_len": 120,
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
    # policy = TanhGaussianBCPolicy(**policy_configs)
    iql_policy = IQLModel(**policy_configs)
    policy = train(policy=policy,
                   dataloader=dataloader,
                   benchmark=benchmark,
                   **train_configs)

    # final_policy = copy.deepcopy(policy)
    # torch.save(final_policy, os.path.join(dest_dir, f"{exp_name}-last.pt"))
    # success_rate, _ = benchmark.evaluate(
    #     policy=final_policy,
    #     render=False,
    #     **eval_configs
    # )
    # print(f"final success_rate: {success_rate}")
    # wandb.log({"final_success_rate": success_rate})

    # best_policy = torch.load(os.path.join(dest_dir, f"{exp_name}-best.pt"))
    # success_rate, _ = benchmark.evaluate(
    #     policy=best_policy,
    #     render=False,
    #     **eval_configs
    # )
    # print(f"best success_rate: {success_rate}")
    # wandb.log({"best_success_rate": success_rate})

    '''generate sample policy rollouts'''
    success_rate, gifs = benchmark.evaluate(
        policy=policy,
        render=True,
        **rollout_configs
    )
    dump_video_wandb(gifs, "rollouts")
