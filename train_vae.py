import cv2
import time
import databoost
# from databoost.models.bc import TanhGaussianBCPolicy
from models.metaworld import TanhGaussianBCPolicy, VAE
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from databoost.base import DataBoostBenchmarkBase


def train_vae(vae: nn.Module,
          dataloader: DataLoader,
          n_steps: int,):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = vae.train().to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=1e-4, betas=(0.9, 0.999))

    step = 0
    epoch = 0
    n_steps = int(n_steps)
    pbar = tqdm(total=n_steps)
    while (step < n_steps):
        epoch += 1
        losses = []
        for _, traj_batch in enumerate(dataloader):
            optimizer.zero_grad()
            obs_batch = traj_batch[0].to(device).float()
            action_batch = traj_batch[1].to(device).float()
            # remove the window dimension, since just 1
            loss = vae.compute_loss(obs_batch, action_batch)
            loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)
            losses.append(loss.item())
            
            if step >= n_steps:
                break
        print(f'epoch{epoch} loss:', np.mean(losses))
        
    pbar.close()
    # np.save(f"{exp_name}_successes.npy", np.array(successes))
    return vae

if __name__ == "__main__":
    task_name = "pick-place-wall"
    goal_conditioned = False
    seq_len = 1
    stride = 1
    mask_goals=True

    # initialize environment
    benchmark = databoost.get_benchmark(benchmark_name='metaworld', mask_goal_pos=mask_goals)
    env = benchmark.get_env(task_name)

    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import NDArrayDecoder
    from ffcv.transforms import ToTensor, ToDevice
    pipelines={
        'observations': [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda'))],
        'actions': [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda'))],
    }

    
    # create policy
    vae_configs = {
        "obs_dim": 78 if goal_conditioned else 39,
        "act_dim": 4,
        "latent_dim": 64,
        "hidden_sizes": [400, 400, 400], # [100, 100, 100]
    }

    # dataset = env._get_dataset(goal_condition=goal_conditioned, seq_len=seq_len, stride=stride)
    # dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    
    cond = 'none'
    # creating policy for bc similarity
    dataloader = Loader(
                fname=f'/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/cond-{cond}_all_traj/cond-{cond}_all_traj.beton',
                batch_size=1000,
                num_workers=5,
                order=OrderOption.QUASI_RANDOM,
                pipelines=pipelines,
                drop_last=False,
            )
    vae = VAE(**vae_configs)
    vae = train_vae(vae=vae,
                dataloader=dataloader,
                n_steps=2e5,)

    torch.save([vae_configs, vae.state_dict()], f"{cond}_vae.pth")
    kwargs, weights = torch.load(f"{cond}_vae.pth", weights_only=False)
    vae = VAE(**kwargs)
    vae.load_state_dict(weights)
    vae = vae.cuda()

    # # loading and evaluating a policy
    # policy = torch.load("/home/shivin/Desktop/datamodels/data/metaworld/models/sample_trained_policy.pt")