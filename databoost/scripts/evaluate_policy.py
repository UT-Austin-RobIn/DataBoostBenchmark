import random

import torch

import databoost


random.seed(42)


# train/load a policy
task = "door-open"
policy_path = "seed_metaworld_door-open_policy.pt"
policy = torch.load(policy_path)

print(f"evaluating {policy_path} on {task} task")
# initialize appropriate benchmark with corresponding task
benchmark = databoost.get_benchmark("metaworld")
# evaluate the policy using the benchmark
success_rate = benchmark.evaluate(
    task_name=task,
    policy=policy,
    n_episodes=100,
    max_traj_len=500
)
print(f"policy success rate: {success_rate}")