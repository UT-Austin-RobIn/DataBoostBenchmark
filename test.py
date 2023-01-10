import databoost
from databoost.envs.metaworld.utils import load_env_state
import cv2
import pickle

# benchmark = databoost.get_benchmark("metaworld")
# env = benchmark.get_env("pick-place-wall")
# ob = env.reset()
with open("/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/seed/plate-slide-back-side/plate-slide-back-side_5.pkl", "rb") as f:
    env = pickle.load(f)
    print(env.random_init)
# env.env = env_copy
# env = load_env_state(env, state)
cv2.imwrite("plate-slide-back-side_5_test.jpg", env.default_render())
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