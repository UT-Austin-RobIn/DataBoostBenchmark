import os
import functools
from typing import Tuple

import cv2
import numpy as np
import torch.nn as nn
from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS, test_cases_latest_nonoise


def trajectory_generator(env: MujocoEnv,
                         policy: nn.Module,
                         act_noise_pct: float = None,
                         res: Tuple[int] = (640, 480),
                         camera: str = "behindGripper"):
    '''Generates MujocoEnv trajectories given a policy.
    Args:
        env [MujocoEnv]: Meta-world's MujocoEnv
        policy [nn.Module]: policy that returns an action given an
                            observation, with a get_action call
        act_noise_pct [float]: noise as percentage of action space range
        res [Tuple[int]]: resolution of rendered images (default: (640, 480)).
                          if None, will not render image
                          (will return im as None)
        camera [str]: Meta-world camera type (default: "behindGripper"); one of
                      ['corner', 'topview', 'behindGripper', 'gripperPOV']
    Returns:
        generator of tuple (
            obs [np.ndarray]: observation; 39 dimensional array
            act [np.ndarray]: action; 4 dimensional array
            rew [float]: reward; float
            done [bool]: done flag
            info [Dict]: task-specific info
            im [np.ndarray]: rendered image after the step
        )
    '''
    action_space_ptp = env.action_space.high - env.action_space.low
    env.reset()
    env.reset_model()
    obs = env.reset()
    for _ in range(env.max_path_length):
        act = policy.get_action(obs)
        act = np.random.normal(act, act_noise_pct * action_space_ptp)
        obs, rew, done, info = env.step(act)
        im = None
        if res is not None:
            im = env.sim.render(*res,
                                mode='offscreen',
                                camera_name=camera)[:, :, ::-1]
            if camera == "behindGripper":  # this view requires a 180 rotation
                im = cv2.rotate(im, cv2.ROTATE_180)
        yield obs, act, rew, done, info, im


config = [
    # env, action noise pct, cycles, quit on success
    ('assembly-v2', np.zeros(4), 3, True),
    ('basketball-v2', np.zeros(4), 3, True),
    # ('bin-picking-v2', np.zeros(4), 3, True),
    # ('box-close-v2', np.zeros(4), 3, True),
    # ('button-press-topdown-v2', np.zeros(4), 3, True),
    # ('button-press-topdown-wall-v2', np.zeros(4), 3, True),
    # ('button-press-v2', np.zeros(4), 3, True),
    # ('button-press-wall-v2', np.zeros(4), 3, True),
    # ('coffee-button-v2', np.zeros(4), 3, True),
    # ('coffee-pull-v2', np.zeros(4), 3, True),
    # ('coffee-push-v2', np.zeros(4), 3, True),
    # ('dial-turn-v2', np.zeros(4), 3, True),
    # ('disassemble-v2', np.zeros(4), 3, True),
    # ('door-close-v2', np.zeros(4), 3, True),
    # ('door-lock-v2', np.zeros(4), 3, True),
    # ('door-open-v2', np.zeros(4), 3, True),
    # ('door-unlock-v2', np.zeros(4), 3, True),
    # ('hand-insert-v2', np.zeros(4), 3, True),
    # ('drawer-close-v2', np.zeros(4), 3, True),
    # ('drawer-open-v2', np.zeros(4), 3, True),
    # ('faucet-open-v2', np.zeros(4), 3, True),
    # ('faucet-close-v2', np.zeros(4), 3, True),
    # ('hammer-v2', np.zeros(4), 3, True),
    # ('handle-press-side-v2', np.zeros(4), 3, True),
    # ('handle-press-v2', np.zeros(4), 3, True),
    # ('handle-pull-side-v2', np.zeros(4), 3, True),
    # ('handle-pull-v2', np.zeros(4), 3, True),
    # ('lever-pull-v2', np.zeros(4), 3, True),
    # ('peg-insert-side-v2', np.zeros(4), 3, True),
    # ('pick-place-wall-v2', np.zeros(4), 3, True),
    # ('pick-out-of-hole-v2', np.zeros(4), 3, True),
    # ('reach-v2', np.zeros(4), 3, True),
    # ('push-back-v2', np.zeros(4), 3, True),
    # ('push-v2', np.zeros(4), 3, True),
    # ('pick-place-v2', np.zeros(4), 3, True),
    # ('plate-slide-v2', np.zeros(4), 3, True),
    # ('plate-slide-side-v2', np.zeros(4), 3, True),
    # ('plate-slide-back-v2', np.zeros(4), 3, True),
    # ('plate-slide-back-side-v2', np.zeros(4), 3, True),
    # ('peg-insert-side-v2', np.zeros(4), 3, True),
    # ('peg-unplug-side-v2', np.zeros(4), 3, True),
    # ('soccer-v2', np.zeros(4), 3, True),
    # ('stick-push-v2', np.zeros(4), 3, True),
    # ('stick-pull-v2', np.zeros(4), 3, True),
    # ('push-wall-v2', np.zeros(4), 3, True),
    # ('push-v2', np.zeros(4), 3, True),
    # ('reach-wall-v2', np.zeros(4), 3, True),
    # ('reach-v2', np.zeros(4), 3, True),
    # ('shelf-place-v2', np.zeros(4), 3, True),
    # ('sweep-into-v2', np.zeros(4), 3, True),
    # ('sweep-v2', np.zeros(4), 3, True),
    # ('window-open-v2', np.zeros(4), 3, True),
    # ('window-close-v2', np.zeros(4), 3, True),
]

for env, noise, cycles, quit_on_success in config:
    tag = env + '-noise-' + \
        np.array2string(noise, precision=2, separator=',', suppress_small=True)

    policy = functools.reduce(
        lambda a, b: a if a[0] == env else b, test_cases_latest_nonoise)[1]
    env = ALL_ENVS[env]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    writer = writer_for(
        tag, env.metadata['video.frames_per_second'], resolution)
    for _ in range(cycles):
        for r, done, info, img in trajectory_generator(env, policy, noise, resolution, camera):
            if flip:
                img = cv2.rotate(img, cv2.ROTATE_180)
            writer.write(img)
            if quit_on_success and info['success']:
                break

    writer.release()
