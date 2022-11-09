import functools
import os
import pickle
from typing import Dict, Tuple

import cv2
import numpy as np
from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from metaworld.policies.policy import Policy

from databoost.utils.general import AttrDict
from databoost.utils.data import write_h5
import databoost.envs.metaworld.config as config


def trajectory_generator(env: MujocoEnv,
                         policy: Policy,
                         act_noise_pct: float = None,
                         res: Tuple[int] = (640, 480),
                         camera: str = "behindGripper"):
    '''Generates MujocoEnv trajectories given a policy.
    Args:
        env [MujocoEnv]: Meta-world's MujocoEnv
        policy [Policy]: policy that returns an action given an
                            observation, with a get_action call
        act_noise_pct [float]: noise as percentage of action space range
        res [Tuple[int]]: resolution of rendered images (default: (640, 480)).
                          if None, will not render image
                          (will return im as None)
        camera [str]: Meta-world camera type (default: "behindGripper"); one of
                      ['corner', 'topview', 'behindGripper', 'gripperPOV']
    Returns:
        generator of tuple (
            ob [np.ndarray]: observation; 39 dimensional array
            act [np.ndarray]: action; 4 dimensional array
            rew [float]: reward; float
            done [bool]: done flag
            info [Dict]: task-specific info
            im [np.ndarray]: rendered image after the step
        )
    '''
    action_space_ptp = env.action_space.high - env.action_space.low
    if act_noise_pct is None:
        act_noise_pct = np.zeros_like(env.action_space.sample())
    env.reset()
    env.reset_model()
    ob = env.reset()
    for _ in range(env.max_path_length):
        act = policy.get_action(ob)
        act = np.random.normal(act, act_noise_pct * action_space_ptp)
        ob, rew, done, info = env.step(act)
        im = None
        if res is not None:
            im = env.render(offscreen=True, camera_name=camera,
                            resolution=res)[:, :, ::-1]
            if camera == "behindGripper":  # this view requires a 180 rotation
                im = cv2.rotate(im, cv2.ROTATE_180)
        yield ob, act, rew, done, info, im


def add_to_traj(traj: AttrDict,
                ob: np.ndarray,
                act: np.ndarray,
                rew: float,
                done: bool,
                info: Dict,
                im: np.ndarray):
    '''helper function to append a step's results to a trajectory dictionary
    Args:
        traj [AttrDict]: dictionary with keys {obs, acts, rews, dones, infos, ims}
        ob [np.ndarray]: observation; 39 dimensional array
        act [np.ndarray]: action; 4 dimensional array
        rew [float]: reward; float
        done [bool]: done flag
        info [Dict]: task-specific info
        im [np.ndarray]: rendered image after the step
    '''
    traj.obs.append(ob)
    traj.acts.append(act)
    traj.rews.append(rew)
    traj.dones.append(done)
    traj.infos.append(pickle.dumps(info))
    traj.ims.append(im)


def traj_to_numpy(traj: AttrDict):
    '''convert trajectories attributes into numpy arrays
    Args:
        traj [AttrDict]: dictionary with keys {obs, acts, rews, dones, infos, ims}
    Returns:
        traj_numpy [AttrDict]: trajectory dict with attributes as numpy arrays
    '''
    traj_numpy = AttrDict()
    for attr in traj:
        traj_numpy[attr] = np.array(traj[attr])
    return traj_numpy


if __name__ == "__main__":
    for task_name, task_config in config.seed_tasks.items():
        task_config = AttrDict(task_config)
        env = task_config.env()
        # Set necessary env attributes
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        # instantiate expert policy
        policy = task_config.expert_policy()
        # generate specified number of successful demos per seed task
        seed_task_dir = os.path.join(config.seed_dataset_dir, task_name)
        os.makedirs(seed_task_dir, exist_ok=True)
        num_success, num_tries = 0, 0
        while num_success < config.num_seed_demos_per_task:
            traj = AttrDict()
            # initialize empty arrays
            for attr in ["obs", "acts", "rews", "dones", "infos", "ims"]:
                traj[attr] = []
            # generate trajectories using expert policy
            for ob, act, rew, done, info, im in trajectory_generator(env, policy):
                add_to_traj(traj, ob, act, rew, done, info, im)
                # done is always false, as per Meta-world's infinite-horizon MDP paradigm
                if info['success']:
                    num_success += 1
                    traj = traj_to_numpy(traj)
                    filename = f"{task_name}_{num_success}.h5"
                    write_h5(traj, os.path.join(seed_task_dir, filename))
                    break
            num_tries += 1
            print(f"generating {task_name} demos: {num_success}/{num_tries}")
