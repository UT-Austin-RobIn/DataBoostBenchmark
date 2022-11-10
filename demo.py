import random

import cv2

import databoost


def main():
    # list out benchmarks
    print(f"benchmarks: {databoost.benchmarks_list}")
    # choose benchmark
    benchmark = databoost.get_benchmark("metaworld")
    # list out benchmark tasks
    print(f"tasks: {benchmark.tasks_list}")
    # choose task
    task = random.choice(benchmark.tasks_list)
    # instantiate corresponding environment
    env = benchmark.get_env(task)
    # get seed dataset (n_demos <= total seed demos)
    seed_dataset = env.get_seed_dataset(n_demos=5)
    for attr, val in seed_dataset.items():
        print(f"{attr} [{type(val)}]: {val.shape}")
    print(f"num_dones: {sum(seed_dataset.dones)}")
    # get prior dataset (n_demos <= total prior demos)
    prior_dataset = env.get_prior_dataset(n_demos=30)
    for attr, val in prior_dataset.items():
        print(f"{attr} [{type(val)}]: {val.shape}")
    print(f"num_dones: {sum(prior_dataset.dones)}")
    print(f"sum of all rewards: {prior_dataset.rewards.sum()}")
    # policy and video writer for demo purposes
    policy = databoost.envs.metaworld.config.tasks[task].expert_policy()
    writer = cv2.VideoWriter(
        f'{task}_demo.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        env.metadata['video.frames_per_second'],
        (640, 480)
    )
    # interact with environment
    ob = env.reset()
    for step_num in range(env.max_path_length):
        act = policy.get_action(ob)
        ob, rew, done, info = env.step(act)
        print(f"{step_num}: {rew}")
        # should probably standardize render API
        im = env.render(offscreen=True, camera_name="behindGripper",
                        resolution=(640, 480))[:, :, ::-1]
        im = cv2.rotate(im, cv2.ROTATE_180)
        writer.write(im)
    writer.release()

if __name__ == "__main__":
    main()
