import random

import cv2
import torch

import databoost


def print_traj_shapes(traj_batch, prefix=None):
    for attr, val in traj_batch.items():
        if (isinstance(val, dict)):
            if prefix is None:
                print_traj_shapes(val, attr)
            else:
                print_traj_shapes(val, f"{prefix}/{attr}")
            continue
        if prefix is None:
            print(f"{attr} [{type(val)}]: {val.shape}")
        else:
            print(f"{prefix}/{attr} [{type(val)}]: {val.shape}")

def main():
    # list out benchmarks
    print(f"benchmarks: {databoost.benchmarks_list}")
    # choose benchmark
    benchmark = databoost.get_benchmark("metaworld")
    # list out benchmark tasks
    print(f"tasks: {benchmark.tasks_list}")
    # choose task
    task = "pick-place-wall"
    # instantiate corresponding environment
    env = benchmark.get_env(task)
    # get seed dataset (n_demos <= total seed demos)
    print("\n---Seed Dataset---")
    seed_dataset = env.get_seed_dataset(n_demos=5)
    print_traj_shapes(seed_dataset)
    print(f"num_dones: {sum(seed_dataset.dones)}")
    print(f"sum of all rewards: {seed_dataset.rewards.sum()}")
    # alternatively, get seed dataloader
    print("\n---Seed Dataloader---")
    seed_dataloader = env.get_seed_dataloader(n_demos=5, seq_len=10, batch_size=3, shuffle=True)
    for seed_traj_batch in seed_dataloader:
        print_traj_shapes(seed_traj_batch)
        break
    # get prior dataset (n_demos <= total prior demos)
    print("\n---Prior Dataset---")
    prior_dataset = env.get_prior_dataset(n_demos=30)
    print_traj_shapes(prior_dataset)
    print(f"num_dones: {sum(prior_dataset.dones)}")
    print(f"sum of all rewards: {prior_dataset.rewards.sum()}")
    # alternatively, get prior dataloader
    print("\n---Prior Dataloader---")
    prior_dataloader = env.get_prior_dataloader(n_demos=20, seq_len=10, batch_size=3, shuffle=True)
    for prior_traj_batch in prior_dataloader:
        print_traj_shapes(prior_traj_batch)
        break
    # policy and video writer for demo purposes
    print("\n---Online Env Interaction---")
    policy = databoost.envs.metaworld.config.tasks[task].expert_policy()
    writer = cv2.VideoWriter(
        f'{task}_demo.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
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
        im = env.render(camera_name="corner",
                        resolution=(640, 480))[:, :, ::-1]
        im = cv2.rotate(im, cv2.ROTATE_180)
        writer.write(im)
    writer.release()
    # train/load a policy
    policy = torch.load("/data/jullian-yapeter/DataBoostBenchmark/metaworld/models/pick-place-wall/large_seed/metaworld-pick-place-wall-large_seed-goal_cond_False-mask_goal_pos_True-9-last.pt")
    # evaluate the policy using the benchmark
    print("\n---Policy Evaluation---")
    success_rate, gif = benchmark.evaluate(
        task_name=task,
        policy=policy,
        n_episodes=100,
        max_traj_len=500
    )
    print(f"policy success rate: {success_rate}")

if __name__ == "__main__":
    main()
