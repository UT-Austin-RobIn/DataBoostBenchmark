import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as ALL_V2_ENVS

from databoost.base import DataBoostEnvWrapper
import databoost.envs.metaworld.config as const


door_open_env = DataBoostEnvWrapper(
    ALL_V2_ENVS['door-open-v2-goal-observable'](),
    seed_dataset_url="databoost/envs/metaworld/data/seed/door-open/door-open_1.h5",
    prior_dataset_url=""
)

__all__ = [door_open_env]