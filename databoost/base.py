from dataclasses import dataclass
import os
from typing import Dict

import gym

from databoost.utils.data import find_h5, read_h5, concatenate_traj_data


@dataclass
class Task:
    '''Specifies required environment name and data.
    Args:
        env_name [str]: name of required environment
        data: [Dict]: data required to configure an environment with this task
    '''
    env_name: str
    data: Dict


class DataBoostBenchmarkBase:
    '''DataBoostBenchmark is a wrapper to standardize the benchmark across
    environments and tasks. This class includes functionality to load the
    environment and datasets (both seed and prior)
    '''

    def __init__(self, env: gym.Env, task: Task):
        self.env = env
        self.task = task

    def evaluate(self):
        raise NotImplementedError


class DataBoostEnvWrapper(gym.Wrapper):

    def __init__(self,
                 env,
                 prior_dataset_url: str,
                 seed_dataset_url: str):
        super().__init__(env)
        self.prior_dataset_url = prior_dataset_url
        self.seed_dataset_url = seed_dataset_url

    def get_seed_dataset(self, n_demos: int):
        seed_dataset_files = find_h5(self.seed_dataset_url)
        assert len(seed_dataset_files) >= n_demos, \
            f"given n_demos too large. Max is {len(seed_dataset_files)}"
        trajs = [read_h5(seed_dataset_files[i]) for i in range(n_demos)]
        trajs = concatenate_traj_data(trajs)
        return trajs

    def get_prior_dataset(self, n_demos: int):
        pass
