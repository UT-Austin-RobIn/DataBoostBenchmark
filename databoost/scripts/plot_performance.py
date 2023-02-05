import os
import random
import argparse

import torch
import numpy as np

import databoost
from databoost.utils.data import write_json
import wandb

random.seed(42)

policy_dir = "/home/jullian-yapeter/data/DataBoostBenchmark/language_table/models/dummy/separate/All_BigNet"
benchmark = "language_table"
task = "separate"
goal_cond = False
mask_goal_pos = False
exp_name = f"{benchmark}-{task}-{boosting_method}-goal_cond_{goal_condition}-mask_goal_pos_{mask_goal_pos}"

wandb.init(
        resume=exp_name,
        project="boost",
        config=configs,
        dir="/home/jullian-yapeter/tmp",
        entity="clvr",
        notes="",
    )

chkpt_range = range(0, 20000, 1000)
n_episodes = 300

benchmark = databoost.get_benchmark(benchmark)
policy_filenames = os.listdir(policy_dir)
success_rates = []

for idx in chkpt_range:
    chkpt = int(idx)
    policy_filename = None
    for fn in policy_filenames:
        if f"-{chkpt}.pt" in fn:
            policy_filename = os.path.join(policy_dir, fn)
            break
    if policy_filename is None:
        print(f"chkpt {chkpt} not found")
        raise ValueError
    print(f"evaluating {policy_filename} on {task} task")
    policy = torch.load(policy_filename)
    # initialize appropriate benchmark with corresponding task
    # evaluate the policy using the benchmark
    success_rate, gif = benchmark.evaluate(
        task_name=task,
        policy=policy,
        n_episodes=n_episodes,
        max_traj_len=120,
        render=False,
        goal_cond=goal_cond
    )
    print(f"policy {idx} success rate: {success_rate}")
    # success_rates.append(success_rate)
    wandb.log({"step": idx, "success_rate": success_rate})
# print(f"avg success: {np.mean(success_rates)}")
# metrics = {
#     "window": n_window,
#     "period": n_period,
#     "n_episodes": n_episodes,
#     "max": np.max(success_rates),
#     "min": np.min(success_rates),
#     "mean": np.mean(success_rates)
# }
# write_json(metrics, os.path.join(policy_dir, f"metrics-chkpt_{int(n_chkpt)}-window_{int(n_window)}-per_{int(n_period)}.json"))
