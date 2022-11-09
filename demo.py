import random

import cv2

import databoost


def main():
    # list out benchmarks
    print(databoost.benchmarks_list)
    # choose benchmark
    benchmark = databoost.benchmark["metaworld"]
    # list out benchmark tasks
    print(benchmark.tasks_list)
    # choose task
    task = random.choice(benchmark.tasks_list)
    # instantiate corresponding environment
    env = benchmark.get_env(task)
    # get seed dataset (n_demos <= len(seed_dataset))
    seed_dataset = env.get_seed_dataset(n_demos=5)
    for attr, val in seed_dataset.items():
        print(f"{attr} [{type(val)}]: {val.shape}")
    print(f"num_dones: {sum(seed_dataset.dones)}")
    # # get prior dataset
    # prior_dataset = env.get_prior_dataset(n_demos=200)
    # for attr, val in prior_dataset.items():
    #     print(f"{attr}: {val.shape}")
    # policy and video writer for demo purposes
    policy = databoost.metaworld.config.tasks[task].expert_policy()
    writer = cv2.VideoWriter(
        f'{task_name}_demo.avi',
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
        im = env.render(offscreen=True, camera_name="behindGripper",
                        resolution=(640, 480))[:, :, ::-1]
        im = cv2.rotate(im, cv2.ROTATE_180)
        writer.write(im)
    writer.release()

if __name__ == "__main__":
    main()
