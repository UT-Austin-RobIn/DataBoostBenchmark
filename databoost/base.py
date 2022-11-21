import copy
import os
import pickle
import random
from typing import Dict, List, Tuple

import cv2
import gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from databoost.utils.general import AttrDict
from databoost.utils.data import (
    find_h5, read_h5, write_h5,
    get_start_end_idxs, concatenate_traj_data
)


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


class DataBoostDataset(Dataset):
    def __init__(self, dataset_dir: str, n_demos: int = None, seq_len: int = None):
        self.dataset_dir = dataset_dir
        self.seq_len = seq_len
        file_paths = find_h5(dataset_dir)
        if n_demos is None: n_demos = len(file_paths)
        if self.seq_len is None:
            assert len(file_paths) >= n_demos, \
                f"given n_demos too large. Max is {len(file_paths)}"
            self.paths = random.sample(file_paths, n_demos)
            return
        self.paths = []
        # filter for files that are long enough
        for file_path in file_paths:
            traj_data = read_h5(file_path)
            traj_len = self.get_traj_len(traj_data)
            if traj_len >= seq_len:  # traj must be long enough
                self.paths.append(file_path)
        print(f"{len(self.paths)}/{len(file_paths)} trajectories "
              "are of sufficient length")
        assert len(self.paths) >= n_demos, \
                f"given n_demos too large. Max is {len(self.paths)}"
        self.paths = random.sample(self.paths, n_demos)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        traj_seq = AttrDict()
        traj_data = read_h5(self.paths[idx])
        if self.seq_len is None:
            return traj_data
        traj_len = self.get_traj_len(traj_data)
        start_end_idxs = get_start_end_idxs(traj_len, self.seq_len)
        slice_start_idx, slice_end_idx = random.choice(start_end_idxs)
        for attr in traj_data:
            if type(traj_data[attr]) in [torch.Tensor, np.ndarray] and traj_data[attr].shape[0] == traj_len:
                traj_seq[attr] = copy.deepcopy(traj_data[attr][slice_start_idx: slice_end_idx])
                assert len(traj_seq[attr]) == self.seq_len
            else:
                # attributes of the trajectory that are not meant to be sliced are simply assigned to each subtrajectory
                traj_seq[attr] = copy.deepcopy([traj_data[attr] for _ in range(self.seq_len)])
        return traj_seq

    def get_traj_len(self, traj_data: AttrDict) -> int:
        '''Get length of trajectory given the AttrDict of trajectory data
        Args:
            traj_data [AttrDict]: trajectory data
        Returns:
            traj_len [int]: length of trajectory
        '''
        return len(traj_data.observations)


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
        if n_demos is None: n_demos = len(dataset_files)
        assert len(dataset_files) >= n_demos, \
            f"given n_demos too large. Max is {len(dataset_files)}"
        rand_idxs = random.sample(range(len(dataset_files)), n_demos)
        trajs = [read_h5(dataset_files[i]) for i in rand_idxs]
        trajs = concatenate_traj_data(trajs)
        return trajs

    def _get_dataloader(self,
                        dataset_dir: str,
                        n_demos: int = None,
                        seq_len: int = None,
                        batch_size: int = 1,
                        shuffle: bool = True):
        dataset = DataBoostDataset(dataset_dir, n_demos, seq_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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

    def get_seed_dataloader(self,
                            n_demos: int = None,
                            seq_len: int = None,
                            batch_size: int = 1,
                            shuffle: bool = True):
        '''loads offline seed dataset corresponding to this environment & task
        Args:
            n_demos [int]: number of demos from dataset to load (if None, load all)
        Returns:
            trajs [AttrDict]: dataset as an AttrDict
        '''
        assert self.seed_dataset_url is not None
        return self._get_dataloader(self.seed_dataset_url,
                                    n_demos=n_demos,
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    shuffle=shuffle)

    def get_prior_dataloader(self,
                             n_demos: int = None,
                             seq_len: int = None,
                             batch_size: int = 1,
                             shuffle: bool = True):
        '''loads offline prior dataset corresponding to this environment
        Args:
            n_demos [int]: number of demos from dataset to load (if None, load all)
        Returns:
            trajs [AttrDict]: dataset as an AttrDict
        '''
        assert self.prior_dataset_url is not None
        return self._get_dataloader(self.prior_dataset_url,
                                    n_demos=n_demos,
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    shuffle=shuffle)


class DatasetGenerationPolicyBase:
    def __init__(self, **datagen_kwargs):
        self.datagen_kwargs = AttrDict(datagen_kwargs)

    def get_action(self, ob: np.ndarray):
        raise NotImplementedError


class DatasetGeneratorBase:
    '''Base dataset generator for all offline DataBoost benchmarks
    '''
    def __init__(self, **dataset_kwargs):
        self.dataset_kwargs = AttrDict(dataset_kwargs)
        self.traj_keys = [
            "observations",
            "actions",
            "rewards",
            "dones",
            "infos",
            "imgs"
        ]

    def init_env(self, task_config):
        raise NotImplementedError

    def init_policy(self, env, task_config):
        raise NotImplementedError

    def get_max_traj_len(self, env, task_config):
        raise NotImplementedError

    def render_img(self, env):
        raise NotImplementedError

    def is_success(self, env, ob, rew, done, info):
        raise NotImplementedError

    def post_process_step(self, env, ob, rew, done, info):
        return ob, rew, done, info

    def trajectory_generator(self,
        env: gym.Env,
        policy: DatasetGenerationPolicyBase,
        task_config,
        do_render: bool = True):
        '''Generates MujocoEnv trajectories given a policy.
        Args:
            env [MujocoEnv]: Meta-world's MujocoEnv
            policy [Policy]: policy that returns an action given an
                                observation, with a get_action call
            do_render [bool]: if true, render images and store it as part of the
                           h5 dataset (render_img function must be overloaded)
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
        task_config = copy.deepcopy(task_config)
        ob = env.reset()
        for _ in range(self.get_max_traj_len(env, task_config)):
            act = policy.get_action(ob)
            nxt_ob, rew, done, info = env.step(act)
            im = None
            if do_render:
                im = self.render_img(env)
            yield ob, act, rew, done, info, im
            ob = nxt_ob

    def init_traj(self):
        traj = AttrDict()
        for attr in self.traj_keys:
            traj[attr] = [] if attr != "infos" else {}
        return traj

    def add_to_traj(self,
        traj: AttrDict,
        ob: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        info: Dict,
        im: np.ndarray = None):
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
        for attr in info:
            if attr not in traj.infos:
                traj.infos[attr] = []
            traj.infos[attr].append(info[attr])
        if im is not None:
            traj.imgs.append(im)

    def traj_to_numpy(self, traj: AttrDict):
        '''convert trajectories attributes into numpy arrays
        Args:
            traj [AttrDict]: dictionary with keys {obs, acts, rews, dones, infos, ims}
        Returns:
            traj_numpy [AttrDict]: trajectory dict with attributes as numpy arrays
        '''
        traj_numpy = self.init_traj()
        for attr in traj:
            if attr != "infos":
                traj_numpy[attr] = np.array(traj[attr])
            else:
                for info_attr in traj.infos:
                    traj_numpy.infos[info_attr] = np.array(traj.infos[info_attr])
        return traj_numpy

    def generate_dataset(self,
        tasks: Dict[str, AttrDict],
        dest_dir: str,
        n_demos_per_task: int,
        mask_reward: bool,
        do_render: bool = True):
        '''generates a dataset given a list of tasks and other configs.

        Args:
            tasks_list [List[str]]: list of task names for which to generate data
            dest_dir [str]: path to directory to which the dataset is to be written
            n_demos_per_task [int]: number of demos to generate per task
            mask_reward [bool]: if true, all rewards are set to zero (for prior dataset)
        '''
        tasks = copy.deepcopy(tasks)
        for task_name, task_config in tasks.items():
            # Initialize env and set necessary env attributes
            task_config = copy.deepcopy(task_config)
            env = self.init_env(task_config)
            # instantiate expert policy
            policy = self.init_policy(env, task_config)
            # generate specified number of successful demos per seed task
            task_dir = os.path.join(dest_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            num_success, num_tries = 0, 0
            while num_success < n_demos_per_task:
                traj = self.init_traj()
                # generate trajectories using expert policy
                for ob, act, rew, done, info, im in self.trajectory_generator(env, policy, task_config, do_render):
                    if mask_reward: rew = 0.0
                    ob, rew, done, info = self.post_process_step(env, ob, rew, done, info)
                    self.add_to_traj(traj, ob, act, rew, done, info, im)
                    if self.is_success(env, ob, rew, done, info):
                        num_success += 1
                        traj = self.traj_to_numpy(traj)
                        filename = f"{task_name}_{num_success}.h5"
                        write_h5(traj, os.path.join(task_dir, filename))
                        break
                num_tries += 1
                print(f"generating {task_name} demos: {num_success}/{num_tries}")
