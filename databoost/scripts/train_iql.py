# import argparse
import clip
import os
import sys
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
from databoost.models.iql.policies import GaussianPolicy, ConcatMlp
from databoost.models.iql.iql import IQLModel


# np.random.seed(42)
# random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(iql_policy,
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
          max_global_steps: int, 
          n_save_best: int,
          goal_condition: bool = False):

    best_success_rate = [-1 for i in range(3)]
    min_success_best = 0
    global_step = iql_policy._n_train_steps_total
    logs, agg = {}, {}
    for epoch in range(int(n_epochs)):
        for cycle in range(n_epoch_cycles):
            for batch_num, batch in tqdm(enumerate(dataloader)):
                log = iql_policy.train_from_torch(batch)
                
                for k in log:
                    if 'losses/' == k[:7] or 'values/' == k[:7]:
                        logs[k] = logs[k]+(log[k]/2000) if k in logs else log[k]/2000
                    else:
                        parent = k.split('/')[0]
                        agg[k] = agg[k] + log[parent + '/num_samples'] if k in agg else log[parent + '/num_samples']
                        logs[k] = logs[k] + log[k] if k in logs else log[k]
                
                global_step += 1

                # if global_step % eval_period == 0:
                #     print(f"evaluating epoch {epoch} with {eval_episodes} episodes")
                #     success_rate, gifs = benchmark.evaluate(
                #         task_name=task_name,
                #         policy=iql_policy,
                #         n_episodes=eval_episodes,
                #         max_traj_len=max_traj_len,
                #         goal_cond=goal_condition,
                #         render=False
                #     )

                #     wandb.log({"success_rate": success_rate}, step=global_step)
                #     print(f"epoch {epoch}: success_rate = {success_rate}")

                #     # saving the top 'n_save_best' policies
                #     if success_rate >= best_success_rate[min_success_best]:
                #         torch.save(iql_policy.get_all_state_dicts(), os.path.join(dest_dir, f"{exp_name}-best{min_success_best}.pt"))
                #         best_success_rate[min_success_best] = success_rate
                #         min_success_best = np.argmin(best_success_rate)

                #     torch.save(iql_policy.get_all_state_dicts(), os.path.join(dest_dir, f"{exp_name}-last.pt"))

                if global_step % 2000 == 0:
                    torch.save(iql_policy.get_all_state_dicts(), os.path.join(dest_dir, f"{exp_name}-step{global_step}.pt"))

                    for k in logs:
                        if ('values_seed/' == k[:12] or 'values_prior/' == k[:13]) and agg[k] > 0:
                            logs[k] /= agg[k]

                    logs['values_seed/num_samples'] = agg['values_seed/num_samples']/(agg['values_seed/num_samples'] + agg['values_prior/num_samples'])
                    logs['values_prior/num_samples'] = 1 - logs['values_seed/num_samples']
                    logs['epochs'] = epoch*n_epoch_cycles + cycle
                    wandb.log(logs, step=global_step)
                    logs, agg = {}, {}

                if global_step >= max_global_steps:
                    break

            if global_step >= max_global_steps:
                break
        if global_step >= max_global_steps:
            break

    return iql_policy

if __name__ == "__main__":
    import databoost
    from databoost.models.bc import BCPolicy, TanhGaussianBCPolicy

    benchmark_name = "language_table"
    task_name = "separate"
<<<<<<< HEAD
    boosting_method = "bc"
=======
    boosting_method = "action"
>>>>>>> 52d374267afc0a7c4936f06acee87a4ef5b0d39f
    goal_condition = False
    mask_goal_pos = False
    exp_name = f"{benchmark_name}-{task_name}-{boosting_method}-{sys.argv[1]}"
    dest_dir = f"/data/sdass/DataBoostBenchmark/{benchmark_name}/models/dummy/{task_name}/{boosting_method}/{sys.argv[1]}/"

    benchmark_configs = {
        "benchmark_name": benchmark_name,
    }

    dataloader_configs = {
        "dataset_dir": [
            # "/data/jullian-yapeter/old_boosted_data/separate/Action/part_0/data/seed/",
            # "/data/jullian-yapeter/old_boosted_data/separate/Action/part_0/data/retrieved/",
            # "/data/jullian-yapeter/old_boosted_data/separate/Action/part_1/data/retrieved/",
            # "/data/jullian-yapeter/old_boosted_data/separate/Action/part_2/data/retrieved/",

            # "/data/jullian-yapeter/old_boosted_data/separate/CLIPLang/part_0/data/seed/",
            # "/data/jullian-yapeter/old_boosted_data/separate/CLIPLang/part_0/data/retrieved/",
            # "/data/jullian-yapeter/old_boosted_data/separate/CLIPLang/part_1/data/retrieved/",
            # "/data/jullian-yapeter/old_boosted_data/separate/CLIPLang/part_2/data/retrieved/",
            # "/data/jullian-yapeter/old_boosted_data/separate/CLIPLang/part_3/data/retrieved/",
            # "/data/jullian-yapeter/old_boosted_data/separate/CLIPLang/part_4/data/retrieved/"

            "/data/jullian-yapeter/old_boosted_data/separate/R3M/part_0/data/seed/",
            # "/data/jullian-yapeter/old_boosted_data/separate/R3M/part_0/data/retrieved/",
            # "/data/jullian-yapeter/old_boosted_data/separate/R3M/part_1/data/retrieved/",
            # "/data/jullian-yapeter/old_boosted_data/separate/R3M/part_2/data/retrieved/",
            # "/data/jullian-yapeter/old_boosted_data/separate/R3M/part_3/data/retrieved/",
            # "/data/jullian-yapeter/old_boosted_data/separate/R3M/part_4/data/retrieved/"

            "/data/jullian-yapeter/old_boosted_data/separate/Handcraft/data/"

            # "/home/jullian-yapeter/data/boosted_data/language_table/Seed",

            # "/home/karl/data/language_table/seed_separate_wider_support",
            # "/home/karl/data/language_table/prior_data_clip",
            # "/data/karl/data/table_sim/rollout_data",
            "/data/jullian-yapeter/DataBoostBenchmark/language_table/boosted_data/separate/BC_5parts_100xeach/part_1/data/seed/",
            "/data/jullian-yapeter/DataBoostBenchmark/language_table/boosted_data/separate/BC_5parts_100xeach/part_1/data/retireved/",
            "/data/jullian-yapeter/DataBoostBenchmark/language_table/boosted_data/separate/BC_5parts_100xeach/part_2/data/retrieved/",
            "/data/jullian-yapeter/DataBoostBenchmark/language_table/boosted_data/separate/BC_5parts_100xeach/part_3/data/retrived/",


            # '/home/sdass/boosting/data/langtable/seed/',
            # '/home/sdass/boosting/data/langtable/retrieved/',
        ],
        
        
        "n_demos": None,
        "batch_size": 128,
        "seq_len": 2,
        "shuffle": True,
        "load_imgs": False,
        "goal_condition": goal_condition,
        "seed_sample_ratio": 0.1,
        "terminal_sample_ratio": 0.01,
        "limited_cache_size": 500000,
    }

    obs_dim = 2048 + 512
    action_dim = 2
    qf_kwargs = dict(hidden_sizes=[512]*3, layer_norm=True)

    policy_configs = {
        "policy": GaussianPolicy(
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        hidden_sizes=[1024]*3,
                        max_log_std=0,
                        min_log_std=-6,
                        std_architecture="values",
                        hidden_activation=nn.LeakyReLU(),
                        # output_activation=nn.Identity(),
                        layer_norm=True,
                    ).to(device),
        # "policy": TanhGaussianBCPolicy(
        #                 env_spec = AttrDict({
        #                     "observation_space": AttrDict({
        #                         "flat_dim": 2048 + 512
        #                     }),
        #                     "action_space": AttrDict({
        #                         "flat_dim": 2
        #                     })
        #                 }),
        #                 hidden_sizes = [512, 512, 512, 512],
        #                 hidden_nonlinearity= nn.LeakyReLU(),
        #                 output_nonlinearity= None,
        #                 min_std =  np.exp(-20.),
        #                 max_std = np.exp(2.)
        #             ).to(device),
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

        "discount": 0.99,
        "quantile": 0.8,
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
        "eval_period": 1e5,
        "eval_episodes": 20,
        "max_traj_len": 60,
        # "n_steps": 5e5,
        "n_epochs": 10000,
        "n_epoch_cycles": 5000,
        "n_save_best": 3,
        "max_global_steps": 5e5,
        "goal_condition": goal_condition
    }

    eval_configs = {
        "task_name": task_name,
        "n_episodes": 50,
        "max_traj_len": 60,
        "goal_cond": goal_condition,
    }

    rollout_configs = {
        "task_name": task_name,
        "n_episodes": 10,
        "max_traj_len": 60,
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
        dir="/home/sdass/tmp",
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
    policy = train(iql_policy=iql_policy,
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
