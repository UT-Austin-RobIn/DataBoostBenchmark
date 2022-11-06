from typing import Dict

from dataclasses import dataclass


@dataclass
class Task:
    '''Specifies required environment name and data.
    Args:
        env_name [str]: name of required environment
        data: [Dict]: data required to configure an environment with this task
    '''
    env_name: str
    data: Dict


@dataclass
class DataConfig:
    '''Specifies configs of returned dataset.
    '''
    size: int


class DataBoostBenchmarkBase:
    '''DataBoostBenchmark is a wrapper to standardize the benchmark across
    environments and tasks. This class includes functionality to load the
    environment and datasets (both seed and prior)
    '''

    def __init__(self, name: str):
        self.name = name

    def get_env(self, task: Task):
        raise NotImplementedError

    def get_seed_dataset(self, config: DataConfig):
        raise NotImplementedError

    def get_prior_dataset(self, config: DataConfig):
        raise NotImplementedError
