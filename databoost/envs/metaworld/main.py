import metaworld

from databoost.base import DataBoostBenchmarkBase, DataConfig
import databoost.envs.metaworld.constants as const


class DataBoostBenchmarkMetaworld(DataBoostBenchmarkBase):
    def __init__(self):
        super().__init__(const.name)

    def get_env(self, task):
        pass

    def get_seed_dataset(self, config: DataConfig):
        raise NotImplementedError

    def get_prior_dataset(self, config: DataConfig):
        raise NotImplementedError
