import databoost
import cv2

benchmark = databoost.get_benchmark("metaworld")
env = benchmark.get_env("assembly")
ob = env.reset()
state = env.get_env_state()
print(ob)
'''
dataset = env.get_prior_dataloader(seq_len=1, goal_condition=True)
i = 0
for traj in dataset:
    # print(traj["imgs"][0][0].shape)
    # print(goal["imgs"][0][0].shape)
    cv2.imwrite(f"traj_{i}.jpg", traj["imgs"][0][0].numpy())
    # cv2.imwrite(f"goal_{i}.jpg", goal["imgs"][0][0].numpy())
    if i > 10: break
    i += 1
'''