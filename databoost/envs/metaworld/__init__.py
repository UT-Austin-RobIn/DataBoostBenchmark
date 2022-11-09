import metaworld

from databoost.base import DataBoostEnvWrapper, DataBoostBenchmarkBase
import databoost.envs.metaworld.config as cfg


class DataBoostBenchmarkMetaworld(DataBoostBenchmarkBase):
    '''Meta-world DataBoost benchmark.
    '''
    def __init__(self):
        self.tasks_list = tasks_list = list(cfg.tasks.keys())

    def get_env(self, task_name: str):
        task_cfg = cfg.tasks.get(task_name)
        assert task_cfg is not None, f"{task_name} is not a valid task name."
        return DataBoostEnvWrapper(
            task_cfg.env(),
            seed_dataset_url=task_cfg.seed_dataset,
            prior_dataset_url=cfg.prior_dataset_dir
        )


__all__ = [DataBoostBenchmarkMetaworld]
