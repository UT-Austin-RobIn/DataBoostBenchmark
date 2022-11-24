import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


random.seed(42)


def train(model: nn.Module, dataloader: DataLoader, n_epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.train().to(device)
    optimizer = optim.RAdam(model.parameters())
    for epoch in range(int(n_epochs)):
        losses = []
        for batch_num, traj_batch in enumerate(dataloader):
            optimizer.zero_grad()
            obs_batch = traj_batch["observations"].to(device)
            obs_batch = obs_batch[:, 0, :]  # remove the window dimension, since just 1
            pred_action_dist = model(obs_batch.float())
            action_batch = traj_batch["actions"].to(device)
            action_batch = action_batch[:, 0, :]  # remove the window dimension, since just 1
            loss = model.loss(pred_action_dist, action_batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: loss = {np.mean(losses)}")
    return model

if __name__ == "__main__":
    import databoost
    from databoost.models.bc import BCPolicy
    

    benchmark_name = "metaworld"
    task_name = "door-open"

    benchmark = databoost.get_benchmark(benchmark_name)
    env = benchmark.get_env(task_name)
    seed_dataloader = env.get_seed_dataloader(
        batch_size=1,
        seq_len=1,
        shuffle=True
    )
    # prior_dataloader = env.get_prior_dataloader(
    #     batch_size=1,
    #     seq_len=1,
    #     shuffle=True
    # )
    
    seed_policy = BCPolicy(
        obs_dim=39,
        action_dim=4,
        hidden_dim=64,
        n_hidden_layers=6
    )
    # prior_policy = BCPolicy(
    #     obs_dim=39,
    #     action_dim=4,
    #     hidden_dim=64,
    #     n_hidden_layers=6
    # )

    seed_policy = train(seed_policy, seed_dataloader, n_epochs=1e3)
    torch.save(seed_policy, f"seed_{benchmark_name}_{task_name}_policy.pt")

    # prior_policy = train(prior_policy, prior_dataloader, n_epochs=1e2)
    # torch.save(prior_policy, f"prior_{benchmark_name}_policy.pt")

    # prior_policy = torch.load(f"prior_{benchmark_name}_policy.pt")
    # finetuned_prior_policy = train(prior_policy, seed_dataloader, n_epochs=500)
    # torch.save(finetuned_prior_policy, f"finetuned_prior_{benchmark_name}_{task_name}_policy.pt")