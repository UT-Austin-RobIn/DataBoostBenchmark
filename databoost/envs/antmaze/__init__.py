import copy
from typing import Dict, List

from d4rl.locomotion.wrappers import NormalizedBoxEnv
import gym
import numpy as np

from databoost.base import DataBoostEnvWrapper, DataBoostBenchmarkBase
from databoost.envs.antmaze.utils import initialize_env, render
import databoost.envs.antmaze.config as cfg


class DataBoostBenchmarkAntMaze(DataBoostBenchmarkBase):

    def __init__(self, eval_mode=True):
        '''AntMaze DataBoost benchmark.

        Args:
            eval_mode [bool]: if False, the environment's observation will include
                              the direction of the goal (as in D4RL)
        Attributes:
            eval [bool]: set to eval_mode argument
            tasks_list [List]: the list of tasks compatible with this benchmark
        '''
        self.eval = eval_mode
        self.tasks_list = list(cfg.tasks.keys())

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
        task_cfg = cfg.tasks.get(task_name)
        assert task_cfg is not None, f"{task_name} is not a valid task name."
        task_cfg = copy.deepcopy(task_cfg)
        env = DataBoostEnvWrapper(
                initialize_env(
                    NormalizedBoxEnv(
                        task_cfg.env(**task_cfg.env_kwargs, eval=self.eval))),
                seed_dataset_url=task_cfg.seed_dataset,
                prior_dataset_url=cfg.prior_dataset_dir,
                render_func=render
            )
        return env

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
        robot = ob[:2]
        dist = np.linalg.norm(robot - env.target_goal)
        success = dist < 0.5
        return success


__all__ = [DataBoostBenchmarkAntMaze]
