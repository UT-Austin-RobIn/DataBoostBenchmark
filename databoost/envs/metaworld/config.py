import os

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as ALL_V2_ENVS

from metaworld.policies import (
    SawyerDoorOpenV2Policy,
    SawyerDoorCloseV2Policy,
    SawyerDoorLockV2Policy,
    SawyerDoorUnlockV2Policy,
)


'''General configs'''
name = "metaworld"
env_root = "/home/jullian-yapeter/code/DataBoostBenchmark/databoost/envs/metaworld"


'''Seed tasks configs'''
seed_dataset_dir = os.path.join(env_root, "data/seed")

num_seed_demos_per_task = 10

seed_tasks = {
    "door-open": {
        "env": ALL_V2_ENVS['door-open-v2-goal-observable'],
        "expert_policy": SawyerDoorOpenV2Policy,
    },
    "door-close": {
        "env": ALL_V2_ENVS['door-close-v2-goal-observable'],
        "expert_policy": SawyerDoorCloseV2Policy,
    },
    "door-lock": {
        "env": ALL_V2_ENVS['door-lock-v2-goal-observable'],
        "expert_policy": SawyerDoorLockV2Policy,
    },
    "door-unlock": {
        "env": ALL_V2_ENVS['door-unlock-v2-goal-observable'],
        "expert_policy": SawyerDoorUnlockV2Policy,
    }
}
