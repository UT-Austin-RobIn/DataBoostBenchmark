from d4rl.locomotion import wrappers

from databoost.base import DataBoostEnvWrapper, DataBoostBenchmarkBase
import databoost.envs.antmaze.config as cfg


class DataBoostBenchmarkAntMaze(DataBoostBenchmarkBase):
    '''D4RL's AntMaze DataBoost benchmark.
    '''
    def __init__(self):
        self.tasks_list = list(cfg.tasks.keys())

    def get_env(self, task_name: str):
        task_cfg = cfg.tasks.get(task_name)
        assert task_cfg is not None, f"{task_name} is not a valid task name."
        return DataBoostEnvWrapper(
            wrappers.NormalizedBoxEnv(task_cfg.env(**task_cfg.env_kwargs)),
            seed_dataset_url=task_cfg.seed_dataset,
            prior_dataset_url=cfg.prior_dataset_dir
        )


__all__ = [DataBoostBenchmarkAntMaze]
