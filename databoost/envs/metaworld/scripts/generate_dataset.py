import functools
import os
import pickle
from typing import Dict, List, Tuple

import cv2
import numpy as np
from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from metaworld.policies.policy import Policy

from databoost.utils.general import AttrDict
from databoost.utils.data import write_h5
import databoost.envs.metaworld.config as cfg
from databoost.envs.metaworld.utils import initialize_env


# keys of trajectory data to be saved for offline use
TRAJ_KEYS = [
    "observations",
    "actions",
    "rewards",
    "dones",
    "infos",
    "imgs"
]


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
        traj [AttrDict]: dictionary with keys {
            observations, actions, rewards, dones, infos, imgs}
        ob [np.ndarray]: observation; 39 dimensional array
        act [np.ndarray]: action; 4 dimensional array
        rew [float]: reward; float
        done [bool]: done flag
        info [Dict]: task-specific info
        im [np.ndarray]: rendered image after the step
    '''
    traj.observations.append(ob)
    traj.actions.append(act)
    traj.rewards.append(rew)
    traj.dones.append(done)
    traj.infos.append(pickle.dumps(info))
    traj.imgs.append(im)


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


def generate_dataset(
    tasks_list: List[str],
    dest_dir: str,
    n_demos_per_task: int,
    act_noise_pct: float,
    resolution: Tuple[int],
    camera: str,
    mask_reward: bool):
    '''generates a metaworld dataset given a list of tasks and other configs.
    
    Args:
        tasks_list [List[str]]: list of metaworld task names for which to generate data;
                                full list is DataBoostBenchmarkMetaworld.tasks_list
        dest_dir [str]: path to directory to which the dataset is to be written
        n_demos_per_task [int]: number of demos to generate per task
        act_noise_pct [float]: noise to add to the action as a percentage
                               of the total action range
        resolution [Tuple[int]]: resolution of rendered images; (width, height)
        camera [str]: Meta-world camera type (default: "behindGripper"); one of
                      ['corner', 'topview', 'behindGripper', 'gripperPOV']
        mask_reward [bool]: if true, all rewards are set to zero (for prior dataset)
    '''
    for task_name in tasks_list:
        task_config = cfg.tasks[task_name]
        env = task_config.env()
        # Set necessary env attributes
        env = initialize_env(env)
        # instantiate expert policy
        policy = task_config.expert_policy()
        # generate specified number of successful demos per seed task
        task_dir = os.path.join(dest_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        num_success, num_tries = 0, 0
        while num_success < n_demos_per_task:
            traj = AttrDict()
            # initialize empty arrays
            for attr in TRAJ_KEYS:
                traj[attr] = []
            # generate trajectories using expert policy
            for ob, act, rew, done, info, im in trajectory_generator(
                env,
                policy,
                act_noise_pct=act_noise_pct,
                res=resolution,
                camera=camera):
                    info.update({
                        "fps": env.metadata['video.frames_per_second'],
                        "resolution": resolution,
                        "act_noise_pct": act_noise_pct
                    })
                    if info['success']: done = True
                    if mask_reward: rew = 0.0
                    add_to_traj(traj, ob, act, rew, done, info, im)
                    # done is always false, as per Meta-world's
                    # infinite-horizon MDP paradigm
                    if info['success']:
                        num_success += 1
                        traj = traj_to_numpy(traj)
                        filename = f"{task_name}_{num_success}.h5"
                        write_h5(traj, os.path.join(task_dir, filename))
                        break
            num_tries += 1
            print(f"generating {task_name} demos: {num_success}/{num_tries}")


if __name__ == "__main__":
    '''Generate seed datasets'''
    generate_dataset(
        tasks_list=cfg.seed_tasks_list,
        dest_dir=cfg.seed_dataset_dir,
        n_demos_per_task=cfg.num_seed_demos_per_task,
        act_noise_pct=cfg.seed_action_noise_pct,
        resolution=cfg.seed_imgs_res,
        camera=cfg.seed_camera,
        mask_reward=False
    )

    '''Generate prior dataset'''
    generate_dataset(
        tasks_list=cfg.prior_tasks_list,
        dest_dir=cfg.prior_dataset_dir,
        n_demos_per_task=cfg.num_prior_demos_per_task,
        act_noise_pct=cfg.prior_action_noise_pct,
        resolution=cfg.prior_imgs_res,
        camera=cfg.prior_camera,
        mask_reward=True
    )