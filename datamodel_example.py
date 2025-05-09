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
    # np.save(f"{exp_name}_successes.npy", np.array(successes))
    return policy

def train2(policy: nn.Module,
          dataloader: DataLoader,
          benchmark: DataBoostBenchmarkBase,
          exp_name: str,
          task_name: str,
          eval_episodes: int,
          eval_max_traj_len: int,
          goal_condition: bool = False,
          **kwargs):
    
    policy.cuda()
    policy.train()
    n_steps = 400

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        betas=(0.9, 0.999),
        weight_decay=1e-5,
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,
        epochs=n_steps,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,
        anneal_strategy='linear',
        div_factor=50,
        final_div_factor=1e4,
    )

    epoch_iter = range(n_steps)
    pbar = tqdm(total=n_steps)
    successes = []
    for epoch in epoch_iter:

        for batch_idx, batch in enumerate(dataloader):

            optimizer.zero_grad(set_to_none=True)

            obs_batch = batch[0]
            action_batch = batch[1]

            out = policy(obs_batch)
            loss = policy.loss(out, action_batch)

            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1, norm_type=2.0)

            loss.backward()
            optimizer.step()
            scheduler.step()
        pbar.update(1)

        if ((epoch+1) % n_steps) == 0:
            print(f"evaluating epoch {epoch} with {eval_episodes} episodes")
            success_rate, _ = benchmark.evaluate(
                policy=policy,
                render=False,
                task_name=task_name,
                n_episodes=eval_episodes,
                max_traj_len=eval_max_traj_len,
                goal_cond=goal_condition
            )

            print(
                f"epoch {epoch}, epoch {epoch}: success_rate = {success_rate}")
            successes.append(success_rate)

    pbar.close()
    np.save(f"{exp_name}_successes.npy", np.array(successes))
    return policy

if __name__ == "__main__":
    task_name = "pick-place-wall"
    goal_conditioned = False
    seq_len = 1
    stride = 1
    mask_goals=True

    # initialize environment
    benchmark = databoost.get_benchmark(benchmark_name='metaworld', mask_goal_pos=mask_goals)
    env = benchmark.get_env(task_name)
    
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
        "obs_dim": 78 if goal_conditioned else 39,
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
        "eval_period": 2e6,
        "eval_episodes": 50,
        "eval_max_traj_len": 500,
        "chkpt_save_period": 5e3,
        "n_steps": 2e5,
        "goal_condition": goal_conditioned
    }

    # top_mask = np.load('/home/shivin/Desktop/datamodels/DataBoostBenchmark/top_residual_masks.npy')
    # pnp_mask = np.load('/home/shivin/Desktop/datamodels/DataBoostBenchmark/pick_place_mask.npy')

    # masks = [top_mask, np.logical_and(top_mask, pnp_mask), None]
    # exp_name = ['dm-residual', 'only-pnp-selected', 'all']
    # masks = [np.logical_and(top_mask, pnp_mask), None, np.logical_and(top_mask, np.logical_not(pnp_mask))]
    # exp_name = ['only-pnp-selected', 'all', 'no-pnp-selected']

    if False:    
        dataset_path = ['/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/cond-last_no-pnp-wall_traj/cond-last_no-pnp-wall_traj.beton']
        masks = [None]
        exp_name = ['all-no-pnp-wall']
        for mask, e, d in zip(masks, exp_name, dataset_path):

            if mask is None:
                indices = None
            else:
                indices = np.where(mask)[0]
            print(indices)
            dataloader = Loader(
                fname=d,
                # batch_size=1000,
                batch_size=16_384,
                num_workers=5,
                order=OrderOption.QUASI_RANDOM,
                pipelines=pipelines,
                drop_last=True,
                indices=indices
            )
            
            # for layers in [3, 2]:
            #     for hidden_size in [400, 100]:
            #         for i in range(3):
            #             policy_configs["hidden_sizes"] = [hidden_size] * layers
            #             train_configs["exp_name"] = f"{e}_{layers}layers_{hidden_size}hidden_{i}"
            #             policy = TanhGaussianBCPolicy(**policy_configs)
            #             policy = train(policy=policy,
            #                         dataloader=dataloader,
            #                         benchmark=benchmark,
            #                         **train_configs)
            for i in range(3):
                train_configs["exp_name"] = f"{e}_{i}"
                policy = TanhGaussianBCPolicy(**policy_configs)
                policy = train2(policy=policy,
                            dataloader=dataloader,
                            benchmark=benchmark,
                            **train_configs)

    dataset = env._get_dataset(goal_condition=goal_conditioned, seq_len=seq_len, stride=stride)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    
    # creating policy for bc similarity
    # dataloader = Loader(
    #             # fname='/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/cond-mw_all_traj/cond-mw_all_traj.beton',
    #             # fname='/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/cond-last_all_traj/cond-last_all_traj.beton',
    #             fname='/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/cond-none_all_traj/cond-none_all_traj.beton',
    #             batch_size=1000,
    #             num_workers=5,
    #             order=OrderOption.QUASI_RANDOM,
    #             pipelines=pipelines,
    #             drop_last=False,
    #         )
    policy = TanhGaussianBCPolicy(**policy_configs)
    policy = train(policy=policy,
                dataloader=dataloader,
                benchmark=benchmark,
                **train_configs)
    
    from training.metaworld import eval_metaworld_sim
    eval_metaworld_sim(policy)

    # torch.save([policy_configs, policy.state_dict()], "none_bc_policy.pth")
    # kwargs, weights = torch.load("none_bc_policy.pth", weights_only=False)
    # policy = TanhGaussianBCPolicy(**kwargs)
    # policy.load_state_dict(weights)
    # policy = policy.cuda()
    
    # success_rate, gif = benchmark.evaluate(
    #     task_name=task_name,
    #     policy=policy,
    #     n_episodes=10,
    #     max_traj_len=500,
    #     goal_cond=goal_conditioned,
    #     render=False
    # )

    # # loading and evaluating a policy
    # policy = torch.load("/home/shivin/Desktop/datamodels/data/metaworld/models/sample_trained_policy.pt")

    # print(gif.shape)
    # for img in gif:
    #     cv2.imshow('x', cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
    #     cv2.waitKey(5)