from dataclasses import dataclass
import os
from typing import Dict

import gym

from databoost.utils.data import read_h5


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

    def get_seed_dataset(self, size: int):
        return read_h5(self.seed_dataset_url)

    def get_prior_dataset(self, size: int):
        pass
