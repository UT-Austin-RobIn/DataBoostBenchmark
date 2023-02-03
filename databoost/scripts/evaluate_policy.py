import os
import random
import argparse

import torch
import numpy as np

import databoost
from databoost.utils.data import write_json


random.seed(42)


# train/load a policy
# parser = argparse.ArgumentParser(
#     description='temp')
# parser.add_argument("--n_chkpt", help="num seed groups")
# parser.add_argument("--n_window", help="num success prior groups")
# args = parser.parse_args()

policy_dir = f"/home/jullian-yapeter/data/DataBoostBenchmark/language_table/models/dummy/separate/OracleNoRL_128"
benchmark = "language_table"
task = "separate"
n_chkpt = int(308000)
n_window = 5
n_period = 2e3
n_episodes = 40

benchmark = databoost.get_benchmark(benchmark)
policy_filenames = os.listdir(policy_dir)
success_rates = []
for idx in range(n_window):
    chkpt = int(n_chkpt - idx * n_period)
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
        max_traj_len=100,
        render=False,
        goal_cond=False
    )
    print(f"policy success rate: {success_rate}")
    success_rates.append(success_rate)
print(f"avg success: {np.mean(success_rates)}")
metrics = {
    "max": np.max(success_rates),
    "min": np.min(success_rates),
    "mean": np.mean(success_rates)
}
write_json(metrics, os.path.join(policy_dir, f"metrics-chkpt_{int(n_chkpt)}-window_{int(n_window)}-per_{int(n_period)}.json"))
