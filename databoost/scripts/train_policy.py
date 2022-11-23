import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train(model: nn.Module, dataloader: DataLoader, n_epochs: int):
    model = model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.RAdam(model.parameters())
    for epoch in range(int(n_epochs)):
        losses = []
        for batch_num, traj in enumerate(dataloader):
            optimizer.zero_grad()
            obs = torch.tensor(traj["observations"]).to(device)
            obs = obs[:, 0]
            pred_action_dist = model(obs.float())
            actions = torch.tensor(traj["actions"]).to(device)
            actions = actions[:, 0]
            loss = pred_action_dist.nll(actions)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: loss = {np.mean(losses)}")
    return model

if __name__ == "__main__":
    import databoost
    from databoost.models.bc import BCPolicy
    

    benchmark = databoost.get_benchmark("metaworld")
    env = benchmark.get_env("door-open")
    seed_dataloader = env.get_seed_dataloader(
        batch_size=1,
        seq_len=1
    )
    
    policy = BCPolicy(
        obs_dim=39,
        action_dim=4,
        hidden_dim=64,
        n_hidden_layers=6
    )

    trained_policy = train(policy, seed_dataloader, n_epochs=1e3)
    torch.save(trained_policy, "my_door_open_policy.pt")