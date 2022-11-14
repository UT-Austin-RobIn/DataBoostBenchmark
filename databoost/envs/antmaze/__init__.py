import copy

from d4rl.locomotion.wrappers import NormalizedBoxEnv

from databoost.base import DataBoostEnvWrapper, DataBoostBenchmarkBase
from databoost.envs.antmaze.utils import initialize_env
import databoost.envs.antmaze.config as cfg


class DataBoostBenchmarkAntMaze(DataBoostBenchmarkBase):
    '''D4RL's AntMaze DataBoost benchmark.
    '''
    def __init__(self, eval_mode=True):
        self.eval = eval_mode
        self.tasks_list = list(cfg.tasks.keys())

    def get_env(self, task_name: str):
        task_cfg = cfg.tasks.get(task_name)
        assert task_cfg is not None, f"{task_name} is not a valid task name."
        task_cfg = copy.deepcopy(task_cfg)
        return DataBoostEnvWrapper(
            initialize_env(
                NormalizedBoxEnv(
                    task_cfg.env(**task_cfg.env_kwargs, eval=self.eval))),
            seed_dataset_url=task_cfg.seed_dataset,
            prior_dataset_url=cfg.prior_dataset_dir
        )


__all__ = [DataBoostBenchmarkAntMaze]
