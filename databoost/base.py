import copy
import os
import pickle
import random
from typing import Callable, Dict, List, Tuple

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from databoost.utils.general import AttrDict
from databoost.utils.data import (
    find_h5, read_h5, write_h5,
    get_start_end_idxs, concatenate_traj_data, get_traj_slice
)


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
                 seed_dataset_url: str,
                 render_func: Callable):
        super().__init__(env)
        self.env = env
        self.prior_dataset_url = prior_dataset_url
        self.seed_dataset_url = seed_dataset_url
        self.render_func = render_func

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

    def default_render(self):
        im = self.render_func(self.env)
        return im


class DataBoostBenchmarkBase:
    def __init__(self):
        '''DataBoostBenchmark is a wrapper to standardize the benchmark across
        environments and tasks. This class includes functionality to load the
        environment and datasets (both seed and prior).

        Attributes:
            tasks_list [List[str]]: list of task names associated with this
                                    benchmark
        '''
        self.tasks_list = None

    def get_env(self, task_name: str) -> DataBoostEnvWrapper:
        '''get the wrapped gym environment corresponding to the specified task.

        Args:
            task_name [str]: the name of the task; must be from the list of
                             tasks compatible with this benchmark (self.tasks_list)
        Returns:
            env [DataBoostEnvWrapper]: wrapped env that implements getters for
                                       the corresponding seed and prior offline
                                       datasets
        '''
        raise NotImplementedError

    def evaluate_success(self,
                         env: gym.Env,
                         ob: np.ndarray,
                         rew: float,
                         done: bool,
                         info: Dict) -> bool:
        '''evaluates whether the given environment step constitutes a success
        in terms of the task at hand. This is used in the benchmark's policy
        evaluator.

        Args:
            env [gym.Env]: gym environment
            ob [np.ndarray]: an observation of the environment this step
            rew [float]: reward received for this env step
            done [bool]: whether the trajectory has reached an end
            info [Dict]: metadata of the environment step
        Returns:
            success [bool]: success flag
        '''
        raise NotImplementedError

    def evaluate(self,
                 task_name: str,
                 policy: nn.Module,
                 n_episodes: int,
                 max_traj_len: int,
                 render: bool = False) -> float:
        '''Evaluates the performance of a given policy on the specified task.

        Args:
            task_name [str]: name of the task to evaluate policy against
            policy [nn.Module]: the policy to evaluate (must implement
                                act = get_action(ob) function)
            n_episodes [int]: number of evaluation episodes
            max_traj_len [int]: max number of steps for one episode
        Returns:
            success_rate [float]: success rate (n_successes/n_episodes)
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        env = self.get_env(task_name)
        policy = policy.eval().to(device)
        n_successes = 0
        if render: gifs = []
        for episode in tqdm(range(int(n_episodes))):
            if render: gif = []
            ob = env.reset()
            if render: gif.append(env.default_render().transpose(2, 0, 1)[::-1])
            for _ in range(max_traj_len - 1):
                with torch.no_grad():
                    act = policy.get_action(ob)
                ob, rew, done, info = env.step(act)
                if render: gif.append(env.default_render().transpose(2, 0, 1)[::-1])
                if self.evaluate_success(env, ob, rew, done, info):
                    n_successes += 1
                    break
            if render and len(gif) > 0:
                if len(gif) < max_traj_len:
                    last_frame = gif[-1]
                    last_frame[1] += 20  # make pad frame green
                    pad = [last_frame for _ in range(max_traj_len - len(gif))]
                    gif += pad
                gifs.append(np.stack(gif))
        success_rate = n_successes / n_episodes
        if render: gifs = np.concatenate(gifs, axis=-1)
        return (success_rate, gifs) if render else success_rate


class DataBoostDataset(Dataset):
    def __init__(self, dataset_dir: str, n_demos: int = None, seq_len: int = None):
        '''DataBoostDataset is a pytorch Dataset class for loading h5-based
        offline trajectory data from a given directory of h5 files.
        Will return AttrDict object where each attribute is of shape:
        (seq_len, *attribute shape).

        Args:
            dataset_dir [str]: path to the directory from which to load h5
                               trajectory data
            n_demos [int]: number of separate h5 files to sample from;
                           if None, sample from all h5 files in the given
                           directory
            seq_len [int]: length of trajectory subsequences to load from files;
                           if None, then will load whole trajectory of each h5
                           file sampled. NOTE: if no seq_len specified, loading
                           batches of size > 1 will result in collate errors
                           since the dimensions will not be equal across
                           trajectories
        '''
        self.dataset_dir = dataset_dir
        self.seq_len = seq_len
        file_paths = find_h5(dataset_dir)
        if n_demos is None: n_demos = len(file_paths)
        if self.seq_len is None:
            # if no seq_len is given, no need to proceed with slicing.
            # use whole trajectories.
            assert len(file_paths) >= n_demos, \
                f"given n_demos too large. Max is {len(file_paths)}"
            self.paths = random.sample(file_paths, n_demos)

            self.slices = []
            for path_id, path in enumerate(self.paths):
                traj_data = read_h5(path)
                traj_len = self.get_traj_len(traj_data)
                self.slices.append((path_id, 0, traj_len))
            print(f"Dataloader contains {len(self.slices)} slices")
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

        self.slices = []
        for path_id, path in enumerate(self.paths):
            traj_data = read_h5(path)
            traj_len = self.get_traj_len(traj_data)
            start_end_idxs = get_start_end_idxs(traj_len, self.seq_len)
            traj_slices = [(path_id, *start_end_idx) for start_end_idx in start_end_idxs]
            self.slices += traj_slices
        print(f"Dataloader contains {len(self.slices)} slices")

    def __len__(self) -> int:
        '''returns length of the dataset; number of traj slices associated with
        this dataset.

        Returns:
            len(self.slices) [int]: number of traj slices that the dataset can sample
        '''
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict:
        '''get item from the dataset; a dictionary of trajectory data.

        Args:
            idx [int]: index of h5 file
        Returns:
            traj_seq [dict]: dictionary of trajectory data, sliced to a random
                             subsequence of specified seq_len; or use whole
                             trajectory if seq_len was not specified
        '''
        path_id, start_idx, end_idx = self.slices[idx]
        traj_data = read_h5(self.paths[path_id])
        if self.seq_len is None:
            return traj_data
        traj_len = self.get_traj_len(traj_data)
        traj_seq = get_traj_slice(traj_data, traj_len, start_idx, end_idx)
        return traj_seq

    def get_traj_len(self, traj_data: Dict) -> int:
        '''Get length of trajectory given the dictionary of trajectory data
        Args:
            traj_data [Dict]: trajectory data
        Returns:
            traj_len [int]: length of trajectory
        '''
        return len(traj_data["observations"])


class DatasetGenerationPolicyBase:
    def __init__(self, **datagen_kwargs):
        '''DataGeneratioPolicyBase standardizes the interface of expert policies
        used to generate offline datasets for each environment. Namely, it
        accepts general datagen_kwargs that are environment and task-specific,
        and enforces a get_action function to be implemented for each child class.

        Args:
            datagen_kwargs [Dict]: env/task-specific configurations to be defined
                                   & used by child classes
        Attributes:
            datagen_kwargs [Dict]: env/task-specific configurations to be defined
                                   & used by child classes
        '''
        self.datagen_kwargs = AttrDict(datagen_kwargs)

    def get_action(self, ob: np.ndarray) -> np.ndarray:
        '''return an action given an observation.

        Args:
            ob [np.ndarray]: an observation from the env step
        Returns:
            act [np.ndarray]: an action determined by the expert policy
        '''
        raise NotImplementedError


class DatasetGeneratorBase:
    def __init__(self, **dataset_kwargs: Dict):
        '''Base dataset generator for all offline DataBoost benchmarks.

        Args:
            dataset_kwargs [Dict]: env/task-specific configurations to be defined
                                   & used by child classes
        Attributes:
            dataset_kwargs [Dict]: env/task-specific configurations to be defined
                                   & used by child classes
            traj_keys [List[str]]: list of attributes in dictionary of 
                                   offline data trajectories
        '''
        self.dataset_kwargs = AttrDict(dataset_kwargs)
        self.traj_keys = [
            "observations",
            "actions",
            "rewards",
            "dones",
            "infos",
            "imgs"
        ]

    def init_env(self, task_config: Dict) -> DataBoostEnvWrapper:
        '''creates an Meta-WOrld environment according to the task specification
        and returns the initialized environment to be used for data collection.

        Args:
            task_config [AttrDict]: contains configs for dataset generation;
                                    importantly, contains task_name, expert_policy
                                    for data collection,and any expert_policy_kwargs.
        Returns:
            env [DataBoostEnvWrapper]: the requested environment
        '''
        raise NotImplementedError

    def init_policy(self,
                    env: gym.Env,
                    task_config: Dict) -> DatasetGenerationPolicyBase:
        '''
        '''
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
