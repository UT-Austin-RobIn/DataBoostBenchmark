import cv2
import time
import databoost
from databoost.models.bc import TanhGaussianBCPolicy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from databoost.base import DataBoostBenchmarkBase


def train(policy: nn.Module,
          dataloader: DataLoader,
          benchmark: DataBoostBenchmarkBase,
          exp_name: str,
          task_name: str,
          dest_dir: str,
          eval_period: int,
          eval_episodes: int,
          eval_max_traj_len: int,
          chkpt_save_period: int,
          n_steps: int,
          goal_condition: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.train().to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.9, 0.999))

    step = 0
    epoch = 0
    losses = []

    n_steps = int(n_steps)
    pbar = tqdm(total=n_steps)
    successes = []
    while (step < n_steps):
        epoch += 1
        for _, traj_batch in enumerate(dataloader):
            optimizer.zero_grad()
            obs_batch = traj_batch[0].to(device)
            # remove the window dimension, since just 1
            pred_action_dist = policy(obs_batch.float())
            action_batch = traj_batch[1].to(device)
            # remove the window dimension, since just 1
            loss = policy.loss(pred_action_dist, action_batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)
            
            # evaluate after every 'eval_period' steps
            if (step % eval_period) == 0:
                if eval_episodes <= 0 or eval_max_traj_len <= 0:
                    print(
                        f"step {step}, epoch {epoch}: loss = {np.mean(losses):.3f}")
                    continue
                print(f"evaluating step {step} with {eval_episodes} episodes")
                success_rate, _ = benchmark.evaluate(
                    policy=policy,
                    render=False,
                    task_name=task_name,
                    n_episodes=eval_episodes,
                    max_traj_len=eval_max_traj_len,
                    goal_cond=goal_condition
                )

                print(
                    f"step {step}, epoch {epoch}: loss = {np.mean(losses):.3f}, success_rate = {success_rate}")
                successes.append(success_rate)
                
                losses = []
            if step >= n_steps:
                break
    pbar.close()
    np.save(f"{exp_name}_successes.npy", np.array(successes))
    return policy

if __name__ == "__main__":
    task_name = "pick-place-wall"
    goal_condition = True

    # initialize environment
    benchmark = databoost.get_benchmark(benchmark_name='metaworld')
    env = benchmark.get_env(task_name)

    # testing if environment is working
    # obs = env.reset()
    # for i in range(100):
    #     obs, reward, done, info = env.step(env.action_space.sample())  
    #     cv2.imshow('x', env.default_render())
    #     cv2.waitKey(1)
    
    # data loader
    # dataloader_configs = {
    #     "batch_size": 1000,
    #     "seq_len": 1,
    #     "shuffle": True,
    #     "load_imgs": False,
    #     "goal_condition": goal_condition
    # }
    # dataloader = env.get_combined_dataloader(**dataloader_configs)

    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import NDArrayDecoder
    from ffcv.transforms import ToTensor, ToDevice
    pipelines={
        'observations': [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda'))],
        'actions': [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda'))],
    }

    
    # create policy
    policy_configs = {
        "obs_dim": 78, # 78
        "act_dim": 4,
        "hidden_sizes": [400, 400, 400], # [100, 100, 100]
        "hidden_nonlinearity": nn.ReLU,
        "output_nonlinearity": None,
        "min_std": np.exp(-20.),
        "max_std": np.exp(2.)
    }

    # training policy
    train_configs = {
        "exp_name": None,
        "task_name": task_name,
        "dest_dir": None,
        "eval_period": 2e4,
        "eval_episodes": 50,
        "eval_max_traj_len": 500,
        "chkpt_save_period": 5e3,
        "n_steps": 2e5,
        "goal_condition": goal_condition
    }

    dataloader = Loader(
        # fname='/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/metaworld_train.beton',
        # fname='/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/hand-curated.beton',
        fname='/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/hand-curated_wo_pick-place-wall.beton',
        batch_size=1000,
        num_workers=5,
        order=OrderOption.QUASI_RANDOM,
        pipelines=pipelines,
        # indices=indices,
        drop_last=True
    )

    # for layers in [2, 3]:
    #     for hidden_size in [100, 200, 400]:
    #         for i in range(3):
    #             policy_configs["hidden_sizes"] = [hidden_size] * layers
    #             train_configs["exp_name"] = f"hand_curated_{layers}layers_{hidden_size}hidden_{i}"
    #             policy = TanhGaussianBCPolicy(**policy_configs)
    #             policy = train(policy=policy,
    #                         dataloader=dataloader,
    #                         benchmark=benchmark,
    #                         **train_configs)

    for i in range(3):    
        train_configs["exp_name"] = f"hand-curated_wo_pick-place-wall_{i}"
        policy = TanhGaussianBCPolicy(**policy_configs)
        policy = train(policy=policy,
                    dataloader=dataloader,
                    benchmark=benchmark,
                    **train_configs)
            
    # train_configs["exp_name"] = f"no_curation"
    # policy = TanhGaussianBCPolicy(**policy_configs)
    # policy = train(policy=policy,
    #             dataloader=dataloader,
    #             benchmark=benchmark,
    #             **train_configs)


    # # loading and evaluating a policy
    # policy = torch.load("/home/shivin/Desktop/datamodels/data/metaworld/models/sample_trained_policy.pt")
    # print(policy)
    # success_rate, gif = benchmark.evaluate(
    #     task_name=task_name,
    #     policy=policy,
    #     n_episodes=10,
    #     max_traj_len=500,
    #     goal_cond=goal_condition,
    #     render=True
    # )

    # print(gif.shape)
    # for img in gif:
    #     cv2.imshow('x', cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
    #     cv2.waitKey(5)