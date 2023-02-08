import copy
import os
import pickle
import random
from typing import Callable, Dict, Tuple

import gym
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset, DataLoader, LTD_CACHE_MAX
)
from tqdm import tqdm

from databoost.utils.general import AttrDict
from databoost.utils.data import (
    find_pkl, find_h5, read_h5,
    write_h5, concatenate_traj_data
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
        render_func [Callable]: environment-specific render function;
                                to create a standardized default_render()
    Attributes:
        env [gym.Env]: instance of Open AI's gym environment
        prior_dataset_url [str]: location of prior dataset
        seed_dataset_url [str]: location of seed dataset
        render_func [Callable]: environment-specific render function;
                                to create a standardized default_render()
    '''

    def __init__(self,
                 env: gym.Env,
                 prior_dataset_url: str,
                 seed_dataset_url: str,
                 render_func: Callable,
                 postproc_func: Callable = None,
                 test_dataset_url: str = None):
        super().__init__(env)
        self.env = env
        self.prior_dataset_url = prior_dataset_url
        self.seed_dataset_url = seed_dataset_url
        self.test_dataset_url = test_dataset_url
        self.render_func = render_func
        self.postproc_func = postproc_func

    def _get_dataset(self, dataset_dir: str, n_demos: int = None) -> AttrDict:
        '''loads offline dataset.
        Args:
            dataset_dir [str]: path to dataset directory
            n_demos [int]: number of demos from dataset to load (if None, load all)
        Returns:
            trajs [AttrDict]: dataset as an AttrDict
        '''
        if type(dataset_dir) in (list, tuple):
            dataset_files = []
            for cur_dataset_dir in dataset_dir:
                dataset_files += find_h5(cur_dataset_dir)
        else:
            dataset_files = find_h5(dataset_dir)
        # if n_demos not specified, use all h5 files in the given dataset dir
        if n_demos is None:
            n_demos = len(dataset_files)
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
                        shuffle: bool = True,
                        load_imgs: bool = True,
                        goal_condition: bool = False,
                        seed_sample_ratio: float = None,
                        terminal_sample_ratio: float = None,
                        limited_cache_size: float = LTD_CACHE_MAX) -> DataLoader:
        '''gets a dataloader to load in h5 data from the given dataset_dir.

        Args:
            dataset_dir [str]: directory of h5 data
            n_demos [int]: the number of demos (h5 files) to retrieve from the
                           dataset dir
            seq_len [int]: the window length with which to split demonstrations
            batch_size [int]: number of sequences to load in as a batch
            shuffle [bool]: shuffle sequences to be loaded
        Returns:
            dataloader [DataLoader]: DataLoader for given dataset directory
        '''
        dataset = DataBoostDataset(
            dataset_dir, n_demos, seq_len,
            load_imgs=load_imgs,
            postproc_func=self.postproc_func,
            goal_condition=goal_condition,
            seed_sample_ratio=seed_sample_ratio,
            terminal_sample_ratio=terminal_sample_ratio,
            limited_cache_size=limited_cache_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_seed_dataset(self, n_demos: int = None) -> Dataset:
        '''loads offline seed dataset corresponding to this environment & task
        Args:
            n_demos [int]: number of demos from dataset to load (if None, load all)
        Returns:
            trajs [AttrDict]: dataset as an AttrDict
        '''
        assert self.seed_dataset_url is not None
        return self._get_dataset(self.seed_dataset_url, n_demos)

    def get_prior_dataset(self, n_demos: int = None) -> Dataset:
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
                            shuffle: bool = True,
                            load_imgs: bool = True,
                            goal_condition: bool = False) -> DataLoader:
        '''gets a dataloader for this benchmark's seed dataset.

        Args:
            n_demos [int]: the number of demos (h5 files) to retrieve from the
                           dataset dir
            seq_len [int]: the window length with which to split demonstrations
            batch_size [int]: number of sequences to load in as a batch
            shuffle [bool]: shuffle sequences to be loaded
        Returns:
            dataloader [DataLoader]: seed DataLoader for this benchmark
        '''
        assert self.seed_dataset_url is not None
        return self._get_dataloader(self.seed_dataset_url,
                                    n_demos=n_demos,
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    load_imgs=load_imgs,
                                    goal_condition=goal_condition)

    def get_prior_dataloader(self,
                             n_demos: int = None,
                             seq_len: int = None,
                             batch_size: int = 1,
                             shuffle: bool = True,
                             load_imgs: bool = True,
                             goal_condition: bool = False) -> DataLoader:
        '''gets a dataloader for this benchmark's prior dataset.

        Args:
            n_demos [int]: the number of demos (h5 files) to retrieve from the
                           dataset dir
            seq_len [int]: the window length with which to split demonstrations
            batch_size [int]: number of sequences to load in as a batch
            shuffle [bool]: shuffle sequences to be loaded
        Returns:
            dataloader [DataLoader]: prior DataLoader for this benchmark
        '''
        assert self.prior_dataset_url is not None
        return self._get_dataloader(self.prior_dataset_url,
                                    n_demos=n_demos,
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    load_imgs=load_imgs,
                                    goal_condition=goal_condition)

    def get_combined_dataloader(self,
                                n_demos: int = None,
                                seq_len: int = None,
                                batch_size: int = 1,
                                shuffle: bool = True,
                                load_imgs: bool = True,
                                goal_condition: bool = False) -> DataLoader:
        '''gets a dataloader for this benchmark's prior dataset.

        Args:
            n_demos [int]: the number of demos (h5 files) to retrieve from the
                           dataset dir
            seq_len [int]: the window length with which to split demonstrations
            batch_size [int]: number of sequences to load in as a batch
            shuffle [bool]: shuffle sequences to be loaded
        Returns:
            dataloader [DataLoader]: prior DataLoader for this benchmark
        '''
        assert self.seed_dataset_url is not None
        assert self.prior_dataset_url is not None
        return self._get_dataloader((self.seed_dataset_url, self.prior_dataset_url),
                                    n_demos=n_demos,
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    load_imgs=load_imgs,
                                    goal_condition=goal_condition)

    def default_render(self) -> np.ndarray:
        '''standard API to wrap environment-specific render function with
        default configurations as set in the passed-in render_func().

        Returns:
            im [np.ndarray]: image rendered with default render function
        '''
        im = self.render_func(self.env)
        return im

    def load_test_env(self, idx=None):
        '''To load saved goal condition test env
        '''
        if self.test_dataset_url is None:
            raise ValueError("this env does not have a corresponding test set")
        test_pkl_files = find_pkl(self.test_dataset_url)
        if idx is not None:
            test_pkl_file = test_pkl_files[idx % len(test_pkl_files)]
        else:
            test_pkl_file = random.choice(test_pkl_files)
        with open(test_pkl_file, "rb") as f:
            test_env_dict = pickle.load(f)
        self.env = test_env_dict["env"]
        test_env_dict.pop("env")
        if self.postproc_func is not None:
            test_env_dict["starting_ob"], _, _, _ = self.postproc_func(
                test_env_dict["starting_ob"], None, None, None)
            test_env_dict["goal_ob"], _, _, _ = self.postproc_func(
                test_env_dict["goal_ob"], None, None, None)
        return test_env_dict

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.postproc_func is not None:
            obs, reward, done, info = self.postproc_func(
                obs, reward, done, info
            )
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.postproc_func is not None:
            obs, _, _, _ = self.postproc_func(
                obs, None, None, None
            )
        return obs


class DataBoostBenchmarkBase:
    def __init__(self, *args):
        '''DataBoostBenchmark is a wrapper to standardize the benchmark across
        environments and tasks. This class includes functionality to load the
        environment and datasets (both seed and prior).

        Attributes:
            tasks_list [List[str]]: list of task names associated with this
                                    benchmark
        '''
        self.tasks_list = None
        self.postproc_func = None

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
                 goal_cond: bool = False,
                 render: bool = False) -> Tuple[float, np.ndarray]:
        '''Evaluates the performance of a given policy on the specified task.

        Args:
            task_name [str]: name of the task to evaluate policy against
            policy [nn.Module]: the policy to evaluate (must implement
                                act = get_action(ob) function)
            n_episodes [int]: number of evaluation episodes
            max_traj_len [int]: max number of steps for one episode
            render [bool]: whether to return gif of eval rollouts
        Returns:
            result [Tuple[float, np.ndarray]]: tuple of success rate
                                               (n_successes/n_episodes) and
                                               gif (None if render is False)
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        env = self.get_env(task_name)
        policy = policy.detach().clone().eval().to(device)
        n_successes = 0
        gifs = [] if render else None
        for episode in tqdm(range(int(n_episodes))):
            if render:
                gif = []
            if not goal_cond:
                ob = env.reset()
            else:
                test_env_dict = env.load_test_env(idx=episode)
                ob = test_env_dict["starting_ob"]
                goal_ob = test_env_dict["goal_ob"]
                # for future use; image based policies
                _ = test_env_dict["goal_im"]
            if render:
                gif.append(env.default_render().transpose(2, 0, 1)[::-1])
            for _ in range(max_traj_len - 1):
                with torch.no_grad():
                    if goal_cond:
                        ob = np.concatenate((ob, goal_ob), axis=-1)
                    act = policy.get_action(ob)

                ob, rew, done, info = env.step(act)
                if render:
                    gif.append(env.default_render().transpose(2, 0, 1)[::-1])
                if self.evaluate_success(env, ob, rew, done, info):
                    n_successes += 1
                    print(n_successes)
                    break
            if render and len(gif) > 0:
                if len(gif) < max_traj_len:
                    last_frame = gif[-1]
                    last_frame[1] += 20  # make pad frame green
                    pad = [last_frame for _ in range(max_traj_len - len(gif))]
                    gif += pad
                gifs.append(np.stack(gif))
        success_rate = n_successes / n_episodes
        if render:
            gifs = np.concatenate(gifs, axis=-1)
        print(f"{n_successes}/{n_episodes} success rate")
        return (success_rate, gifs)


class DataBoostDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 n_demos: int = None,
                 seq_len: int = None,
                 load_imgs: bool = True,
                 postproc_func: Callable = None,
                 goal_condition: bool = False,
                 seed_sample_ratio: float = None,
                 terminal_sample_ratio: float = None,
                 limited_cache_size: int = 10000):
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
        Attributes:
            dataset_dir [str]: path to the directory from which to load h5
                               trajectory data
            seq_len [int]: length of trajectory subsequences to load from files;
                           if None, then will load whole trajectory of each h5
                           file sampled
            paths [List[str]]: list of h5 file paths to load in
            slices [List[Tuple[int]]]: list of tuples (path, start_idx, end_idx)
        '''
        self.limited_cache_size = limited_cache_size
        self.dataset_dir = dataset_dir
        self.seq_len = seq_len
        self.goal_condition = goal_condition
        self.paths = []
        self.path_lens = {}

        if type(dataset_dir) in (list, tuple):
            file_paths = []
            for cur_dataset_dir in dataset_dir:
                file_paths += find_h5(cur_dataset_dir)
        else:
            file_paths = find_h5(dataset_dir)

        self.seed_data = []
        self.seed_lens = 0
        
        self.prior_data = {}
        self.prior_paths = []
        self.prior_lens = 0
        for files in tqdm(file_paths):
            if ('seed' not in files) and len(self.prior_data.keys()) >= self.limited_cache_size:
                self.prior_paths.append(files)
                continue

            traj = read_h5(files)
            traj_len = self.get_traj_len(traj)
            
            for k in  ['file_paths', 'start_end_idxs', 'infos', 'info', 'imgs']:
                if k in traj:
                    del traj[k]
            for k in traj:
                traj[k] = np.concatenate((np.array(traj[k]).reshape(traj_len, -1), np.zeros_like(traj[k][0]).reshape(1, -1)), axis=0)

            traj['dones'] = np.zeros(traj_len+1)
            traj['dones'][-2] = 1.
            traj['rewards'] = np.zeros(traj_len+1)
            if 'Seed' in files or 'seed' in files:   
                traj['rewards'][-2] = 1.
                traj['seed'] = np.ones(traj_len+1)
                
                self.seed_lens += traj_len
                self.seed_data.append(traj)
            else:
                self.prior_paths.append(files)
                traj['seed'] = np.zeros(traj_len+1)
                self.prior_data[files] = traj
                self.prior_lens += traj_len
        
        if len(self.prior_data.keys()) >= self.limited_cache_size:
            self.prior_lens = int(self.prior_lens*len(self.prior_paths)/self.limited_cache_size)

        print("Seed len", self.seed_lens)
        print("Prior len", self.prior_lens)

        self.seed_sample_ratio = self.seed_lens/len(self) if seed_sample_ratio is None else seed_sample_ratio   
        self.terminal_sample_ratio = terminal_sample_ratio*self.seed_sample_ratio if terminal_sample_ratio is not None else None
        
    def __len__(self) -> int:
        '''returns length of the dataset; number of traj slices associated with
        this dataset.

        Returns:
            len(self.slices) [int]: number of traj slices that the dataset can sample
        '''
        return self.seed_lens + self.prior_lens

    def process_prior_file_for_iql(self, files):
        traj = read_h5(files)
        traj_len = self.get_traj_len(traj)

        for k in  ['file_paths', 'start_end_idxs', 'infos', 'info', 'imgs']:
                if k in traj:
                    del traj[k]
        for k in traj:
            traj[k] = np.concatenate((np.array(traj[k]).reshape(traj_len, -1), np.zeros_like(traj[k][0]).reshape(1, -1)), axis=0)        
        traj['dones'] = np.zeros(traj_len+1)
        traj['dones'][-2] = 1
        traj['rewards'] = np.zeros(traj_len+1)
        traj['seed'] = np.zeros(traj_len+1)

        return traj

    def __getitem__(self, idx: int) -> Dict:
        '''get item from the dataset; a dictionary of trajectory data.

        Args:
            idx [int]: index of slice
        Returns:
            traj_seq [dict]: dictionary of trajectory data, sliced to a random
                             subsequence of specified seq_len; or use whole
                             trajectory if seq_len was not specified
        '''
        r = np.random.random()
        if self.prior_lens == 0 or r < self.seed_sample_ratio:
            seq_idx = np.random.randint(len(self.seed_data))
            traj_len = self.get_traj_len(self.seed_data[seq_idx])

            if self.terminal_sample_ratio and r < self.terminal_sample_ratio:
                sample_idx = traj_len - 2
            else:
                sample_idx = np.random.randint(traj_len-1)
            return {
                "observations": self.seed_data[seq_idx]["observations"][sample_idx: sample_idx+2].copy(),
                "actions": (self.seed_data[seq_idx]["actions"][sample_idx: sample_idx+2]).copy(),
                "rewards": self.seed_data[seq_idx]["rewards"][sample_idx: sample_idx+2].copy(),
                "dones": self.seed_data[seq_idx]["dones"][sample_idx: sample_idx+2].copy(),
                "seed": self.seed_data[seq_idx]["seed"][sample_idx: sample_idx+2].copy(),
            }
        else:
            filename = self.prior_paths[np.random.randint(len(self.prior_paths))]

            traj_len = self.get_traj_len(self.prior_data[filename])
            sample_idx = np.random.randint(traj_len-1)
            return {
                "observations": self.prior_data[filename]["observations"][sample_idx: sample_idx+2].copy(),
                "actions": (self.prior_data[filename]["actions"][sample_idx: sample_idx+2]).copy(),
                "rewards": self.prior_data[filename]["rewards"][sample_idx: sample_idx+2].copy(),
                "dones": self.prior_data[filename]["dones"][sample_idx: sample_idx+2].copy(),
                "seed": self.prior_data[filename]["seed"][sample_idx: sample_idx+2].copy(),
            }

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
        '''creates an environment according to the task specification
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
        '''Get an initialized policy for data generation purposes.

        Args:
            env [gym.Env]: the environment with which to generate data
            task_config [Dict]: configs for the policy
        Returns:
            policy [DatasetGenerationPolicyBase]: expert policy to generate data
        '''
        raise NotImplementedError

    def get_max_traj_len(self,
                         env: gym.Env,
                         task_config: Dict) -> int:
        '''get the maximum allowed trajectory length for an episode

        Args:
            env [gym.Env]: the environment with which to generate data
            task_config [Dict]: configs for the data generation process
        Returns:
            max_traj_len [int]: max trajectory length
        '''
        raise NotImplementedError

    def render_img(self, env) -> np.ndarray:
        '''function to render an image of the environment (env-specific)

        Args:
            env [gym.Env]: the environment with which to generate data
        Returns:
            im [np.ndarray]: image of the environment
        '''
        raise NotImplementedError

    def is_success(self,
                   env: gym.Env,
                   ob: np.ndarray,
                   rew: float,
                   done: bool,
                   info: Dict) -> bool:
        '''Determine if the given env step marked a successful episode

        Args:
            env [gym.Env]: gym environment
            ob [np.ndarray]: an observation of the environment this step
            rew [float]: reward received for this env step
            done [bool]: whether the trajectory has reached an end
            info [Dict]: metadata of the environment step
        Returns:
            is_success [bool]: whether the episode was successful
        '''
        raise NotImplementedError

    def get_env_state(self, env):
        '''get state of env such that we can reset to this exact state.
        Complement of load_env_state.
        '''
        raise NotImplementedError

    def post_process_step(self,
                          env: gym.Env,
                          ob: np.ndarray,
                          rew: float,
                          done: bool,
                          info: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        '''Optional post-processing step to apply to each env step in the data
        generation process

        Args:
            env [gym.Env]: gym environment
            ob [np.ndarray]: an observation of the environment this step
            rew [float]: reward received for this env step
            done [bool]: whether the trajectory has reached an end
            info [Dict]: metadata of the environment step
        Returns:
            ob [np.ndarray]: post-processed observation
            rew [float]: post-processed reward
            done [bool]: post-processed done flag
            info [Dict]: post-processed info dict
        '''
        return ob, rew, done, info

    def trajectory_generator(self,
                             env: gym.Env,
                             ob: np.ndarray,
                             policy: DatasetGenerationPolicyBase,
                             task_config,
                             do_render: bool = True):
        '''Generates MujocoEnv trajectories given a policy.
        Args:
            env [MujocoEnv]: Meta-world's MujocoEnv
            ob: [np.ndarray]: starting observation
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
        for _ in range(self.get_max_traj_len(env, task_config)):
            im = None
            if do_render:
                im = self.render_img(env)
            act = policy.get_action(ob)
            nxt_ob, rew, done, info = env.step(act)
            yield ob, act, rew, done, info, im
            ob = nxt_ob

    def init_traj(self) -> Dict:
        '''Initialize an empty trajectory, preparing for data collection

        Returns:
            traj [Dict]: dict of empty attributes
        '''
        traj = AttrDict()
        for attr in self.traj_keys:
            traj[attr] = [] if attr not in ("info", "infos") else {}
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

    def traj_to_numpy(self, traj: AttrDict) -> AttrDict:
        '''convert trajectories attributes into numpy arrays

        Args:
            traj [AttrDict]: dictionary with keys {obs, acts, rews, dones, infos, ims}
        Returns:
            traj_numpy [AttrDict]: trajectory dict with attributes as numpy arrays
        '''
        traj_numpy = self.init_traj()
        for attr in traj:
            if attr not in ("info", "infos"):
                traj_numpy[attr] = np.array(traj[attr])
            else:
                for info_attr in traj.infos:
                    traj_numpy.infos[info_attr] = np.array(
                        traj.infos[info_attr])
        return traj_numpy

    def generate_dataset(self,
                         tasks: Dict[str, AttrDict],
                         dest_dir: str,
                         n_demos_per_task: int,
                         mask_reward: bool,
                         generate_failures: bool = False,
                         do_render: bool = True,
                         save_env_and_goal: bool = False):
        '''generates a dataset given a list of tasks and other configs.

        Args:
            tasks_list [List[str]]: list of task names for which to generate data
            dest_dir [str]: path to directory to which the dataset is to be written
            n_demos_per_task [int]: number of demos to generate per task
            mask_reward [bool]: if true, all rewards are set to zero (for prior dataset)
        '''
        tasks = copy.deepcopy(tasks)
        for task_id, (task_name, task_config) in enumerate(tasks.items()):
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
                ob = env.reset()
                if save_env_and_goal:
                    starting_ob = copy.deepcopy(ob)
                    env_copy = copy.deepcopy(env)
                for ob, act, rew, done, info, im in self.trajectory_generator(
                        env, ob, policy, task_config, do_render):
                    if mask_reward:
                        rew = 0.0
                    ob, rew, done, info = self.post_process_step(
                        env, ob, rew, done, info)
                    self.add_to_traj(traj, ob, act, rew, done, info, im)
                    if not generate_failures and not self.is_success(env, ob, rew, done, info):
                        continue
                    if generate_failures and not (len(traj["observations"]) > 200 and not self.is_success(env, ob, rew, done, info)):
                        continue
                    num_success += 1
                    traj = self.traj_to_numpy(traj)
                    filename = f"{task_name}_{num_success}"
                    if generate_failures:
                        filename += "_fail"
                    traj["dones"][-1] = True
                    if not save_env_and_goal:
                        write_h5(traj, os.path.join(
                            task_dir, filename + ".h5"))
                    else:
                        with open(os.path.join(
                                task_dir, filename + ".pkl"), "wb") as f:
                            goal_ob = traj.observations[-1]
                            goal_im = traj.imgs[-1]
                            pickle.dump({
                                "env": env_copy,
                                "starting_ob": starting_ob,
                                "goal_ob": goal_ob,
                                "goal_im": goal_im
                            }, f)
                    break
                num_tries += 1
                print(
                    f"generating {task_id}; {task_name} demos: {num_success}/{num_tries}")
