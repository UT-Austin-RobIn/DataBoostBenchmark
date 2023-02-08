import os
import numpy as np
import torch

import databoost


# Evaluation configs
benchmark = "metaworld"
task = "pick-place-wall"
n_episodes = 300
max_traj_len = 500
goal_cond = True
render = False
#  Choose policy to evaluate
policy_path = "sample_trained_policy.pt"


benchmark = databoost.get_benchmark(benchmark)
policy = torch.load(policy_path)
print(f"evaluating {policy_path} on {task} task")
# initialize appropriate benchmark with corresponding task
# evaluate the policy using the benchmark
success_rate, gif = benchmark.evaluate(
    task_name=task,
    policy=policy,
    n_episodes=n_episodes,
    max_traj_len=max_traj_len,
    render=render,
    goal_cond=goal_cond
)
print(f"policy success rate: {success_rate}")

if render and gif is not None:
    from PIL import Image
    imgs = [Image.fromarray(img) for img in gif]
    gif_dest_path = policy_path.replace(".pt", ".gif")
    imgs[0].save(gif_dest_path, save_all=True,
                 append_images=imgs[1:], duration=100, loop=0)
