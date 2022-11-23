import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import gym
from tqdm import tqdm


def evaluate(model: nn.Module, env: gym.Env, n_episodes: int, max_traj_len: int):
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_successes = 0
    for episode in tqdm(range(int(n_episodes))):
        ob = env.reset()
        for _ in range(max_traj_len):
            act_dist = model(torch.tensor([ob], dtype=torch.float).to(device))
            act = act_dist.mu.cpu().detach().numpy()[0]
            ob, rew, done, info = env.step(act)
            if info["success"]:
                n_successes += 1
                break
    success_rate = n_successes / n_episodes
    print(f"success rate: {success_rate}")
    return success_rate

if __name__ == "__main__":
    import databoost
    

    benchmark = databoost.get_benchmark("metaworld")
    env = benchmark.get_env("door-open")
    
    policy = torch.load("my_door_open_policy.pt")

    success_rate = evaluate(policy, env, n_episodes=100, max_traj_len=500)
