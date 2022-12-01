import os
import random

import torch

import databoost


random.seed(42)


# train/load a policy
benchmark = "metaworld"
task = "assembly"
policy_dir = f"/data/jullian-yapeter/DataBoostBenchmark/{benchmark}/models"
policy_filename = f"seed_{benchmark}_{task}_policy_3.pt"
policy = torch.load(os.path.join(policy_dir, policy_filename))

print(f"evaluating {policy_filename} on {task} task")
# initialize appropriate benchmark with corresponding task
benchmark = databoost.get_benchmark(benchmark)
# evaluate the policy using the benchmark
success_rate = benchmark.evaluate(
    task_name=task,
    policy=policy,
    n_episodes=100,
    max_traj_len=500
)
print(f"policy success rate: {success_rate}")