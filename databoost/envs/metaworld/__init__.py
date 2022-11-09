import metaworld

from databoost.base import DataBoostEnvWrapper
import databoost.envs.metaworld.config as cfg


tasks_list = list(cfg.tasks.keys())


def get_env(task_name: str):
    task_cfg = cfg.tasks.get(task_name)
    assert task_cfg is not None, f"{task_name} is not a valid task name."
    return DataBoostEnvWrapper(
        task_cfg.env(),
        seed_dataset_url=task_cfg.seed_dataset,
        prior_dataset_url=""
    )


__all__ = [tasks_list, get_env]
