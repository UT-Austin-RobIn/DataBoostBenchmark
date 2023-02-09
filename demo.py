import cv2
from PIL import Image
import numpy as np
import torch

import databoost
from databoost.utils.general import print_traj_shapes


def main():
    # list out benchmarks
    print(f"benchmarks: {databoost.benchmarks_list}")
    # choose benchmark
    benchmark = databoost.get_benchmark("metaworld")
    # list out benchmark tasks: currently each benchmark only supports one task
    # metaworld -> "pick-place-wall"
    # language_table -> "separate"
    print(f"tasks: {benchmark.tasks_list}")
    # choose task
    task = "pick-place-wall"
    # instantiate corresponding environment
    env = benchmark.get_env(task)
    # get seed dataset (n_demos <= total seed demos)
    print("\n---Seed Dataset---")
    # import pdb; pdb.set_trace()
    seed_dataset = env.get_seed_dataset(n_demos=5)
    print_traj_shapes(seed_dataset)
    print(f"num_dones: {sum(seed_dataset.dones)}")
    print(f"sum of all rewards: {seed_dataset.rewards.sum()}")
    # alternatively, get seed dataloader
    print("\n---Seed Dataloader---")
    seed_dataloader = env.get_seed_dataloader(
        n_demos=5, seq_len=10, batch_size=3, shuffle=True)
    for seed_traj_batch in seed_dataloader:
        print_traj_shapes(seed_traj_batch)
        break
    # get prior dataset (n_demos <= total prior demos)
    print("\n---Prior Dataset---")
    prior_dataset = env.get_prior_dataset(n_demos=10)
    print_traj_shapes(prior_dataset)
    print(f"num_dones: {sum(prior_dataset.dones)}")
    print(f"sum of all rewards: {prior_dataset.rewards.sum()}")
    # alternatively, get prior dataloader
    print("\n---Prior Dataloader---")
    prior_dataloader = env.get_prior_dataloader(
        n_demos=30, seq_len=10, batch_size=3, shuffle=True)
    for prior_traj_batch in prior_dataloader:
        print_traj_shapes(prior_traj_batch)
        break
    # get combined seed and prior dataloader
    print("\n---Combined Dataloader---")
    prior_dataloader = env.get_combined_dataloader(
        n_demos=40, seq_len=10, batch_size=3, shuffle=True)
    for prior_traj_batch in prior_dataloader:
        print_traj_shapes(prior_traj_batch)
        break
    # policy and video writer for demo purposes
    print("\n---Online Env Interaction---")
    policy = databoost.envs.metaworld.config.tasks[task].expert_policy()
    writer = cv2.VideoWriter(
        f'{task}_demo.avi',
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        env.metadata['video.frames_per_second'],
        (640, 480)
    )
    # interact with environment
    ob = env.reset()
    for step_num in range(10):
        act = policy.get_action(ob)
        ob, rew, done, info = env.step(act)
        print(f"{step_num}: {rew}")
        # should probably standardize render API
        im = env.default_render()
        writer.write(im)
    writer.release()
    # train/load a policy
    policy = torch.load("CurateDataset/metaworld/models/sample_trained_policy.pt")
    # evaluate the policy using the benchmark
    print("\n---Policy Evaluation---")
    success_rate, gif = benchmark.evaluate(
        task_name=task,
        policy=policy,
        n_episodes=5,
        max_traj_len=500,
        goal_cond=True,
        render=True
    )
    print(f"policy success rate: {success_rate}")
    # import pdb; pdb.set_trace()
    gif = gif.transpose(0, 2, 3, 1)
    imgs = [Image.fromarray(img) for img in gif]
    gif_dest_path = "sample_rollout.gif"
    imgs[0].save(gif_dest_path, save_all=True,
                 append_images=imgs[1:], duration=100, loop=0)


if __name__ == "__main__":
    main()
