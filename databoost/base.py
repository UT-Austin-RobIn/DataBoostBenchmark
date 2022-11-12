import os
import pickle
import random
from typing import Dict, List, Tuple

import cv2
import gym
import numpy as np

from databoost.utils.general import AttrDict
from databoost.utils.data import find_h5, read_h5, write_h5, concatenate_traj_data


class DataBoostBenchmarkBase:
    '''DataBoostBenchmark is a wrapper to standardize the benchmark across
    environments and tasks. This class includes functionality to load the
    environment and datasets (both seed and prior)
    '''
    def __init__(self):
        self.tasks_list = None

    def get_env(self, task_name: str):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class DataBoostEnvWrapper(gym.Wrapper):
    '''DataBoost benchmark's gym wrapper to add offline dataset loading
    capability to gym environments.
    prior_dataset_url and seed_dataset_url must be set for their respective
    get_dataset methods to execute.

    Args:
        env [gym.Env]: instance of Open AI's gym environment
        prior_dataset_url [str]: location of prior dataset
        seed_dataset_url [str]: location of seed dataset
    '''
    def __init__(self,
                 env: gym.Env,
                 prior_dataset_url: str,
                 seed_dataset_url: str):
        super().__init__(env)
        self.prior_dataset_url = prior_dataset_url
        self.seed_dataset_url = seed_dataset_url

    def _get_dataset(self, dataset_dir: str, n_demos: int = None):
        '''loads offline dataset.
        Args:
            dataset_dir [str]: path to dataset directory
            n_demos [int]: number of demos from dataset to load (if None, load all)
        Returns:
            trajs [AttrDict]: dataset as an AttrDict
        '''
        dataset_files = find_h5(dataset_dir)
        assert len(dataset_files) >= n_demos, \
            f"given n_demos too large. Max is {len(dataset_files)}"
        if n_demos is None: n_demos = len(dataset_files)
        rand_idxs = random.sample(range(len(dataset_files)), n_demos)
        trajs = [read_h5(dataset_files[i]) for i in rand_idxs]
        trajs = concatenate_traj_data(trajs)
        return trajs

    def get_seed_dataset(self, n_demos: int = None):
        '''loads offline seed dataset corresponding to this environment & task
        Args:
            n_demos [int]: number of demos from dataset to load (if None, load all)
        Returns:
            trajs [AttrDict]: dataset as an AttrDict
        '''
        assert self.seed_dataset_url is not None
        return self._get_dataset(self.seed_dataset_url, n_demos)

    def get_prior_dataset(self, n_demos: int = None):
        '''loads offline prior dataset corresponding to this environment
        Args:
            n_demos [int]: number of demos from dataset to load (if None, load all)
        Returns:
            trajs [AttrDict]: dataset as an AttrDict
        '''
        assert self.prior_dataset_url is not None
        return self._get_dataset(self.prior_dataset_url, n_demos)


class DatasetGenerationPolicy:
    def get_action(ob: np.ndarray):
        raise NotImplementedError


class DatasetGenerator:
    '''Base dataset generator for all offline DataBoost benchmarks
    '''
    def __init__(self, **dataset_gen_kwargs):
        self.dataset_gen_kwargs = dataset_gen_kwargs
        self.traj_keys = [
            "observations",
            "actions",
            "rewards",
            "dones",
            "infos",
            "imgs"
        ]

    def initialize_env(self, env):
        return env

    def get_max_traj_len(self):
        raise NotImplementedError

    def render_img(self, env):
        raise NotImplementedError

    def trajectory_generator(env: gym.Env,
                             policy: DatasetGenerationPolicy,
                             render: bool,
                             **data_generation_kwargs):
        '''Generates MujocoEnv trajectories given a policy.
        Args:
            env [MujocoEnv]: Meta-world's MujocoEnv
            policy [Policy]: policy that returns an action given an
                                observation, with a get_action call
            render [bool]: if true, render images and store it as part of the
                           h5 dataset (render_img function must be overloaded)
            data_generation_kwargs [Dict]: any other env/task-specific configs
        Returns:
            generator of tuple (
                ob [np.ndarray]: env-specific observation
                act [np.ndarray]: env-specific action
                rew [float]: reward; float
                done [bool]: done flag
                info [Dict]: task-specific info
                im [np.ndarray]: rendered image after the step
            )
        '''
        env = self.initialize_env(env)
        ob = env.reset()
        for _ in range(self.get_max_traj_len()):
            act = policy.get_action(ob)
            ob, rew, done, info = env.step(act)
            im = None
            if render:
                im = self.render_img(env)
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
            ob [np.ndarray]: env-specific observation
            act [np.ndarray]: env-specific action
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
        mask_reward: bool,
        **data_generation_kwargs: Dict):
        '''generates a dataset given a list of tasks and other configs.
        
        Args:
            tasks_list [List[str]]: list of task names for which to generate data
            dest_dir [str]: path to directory to which the dataset is to be written
            n_demos_per_task [int]: number of demos to generate per task
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