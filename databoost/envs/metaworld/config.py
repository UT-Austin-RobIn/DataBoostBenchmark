import os

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as ALL_V2_ENVS

from databoost.utils.general import AttrDict

from metaworld.policies import (
    SawyerDoorOpenV2Policy,
    SawyerDoorCloseV2Policy,
    SawyerDoorLockV2Policy,
    SawyerDoorUnlockV2Policy,
)


'''General configs'''
env_root = "/home/jullian-yapeter/code/DataBoostBenchmark/databoost/envs/metaworld"


'''Tasks configs'''
tasks = {
    "door-open": AttrDict({
        "env": ALL_V2_ENVS['door-open-v2-goal-observable'],
        "seed_dataset": "databoost/envs/metaworld/data/seed/door-open",
        "expert_policy": SawyerDoorOpenV2Policy,
    }),
    "door-close": AttrDict({
        "env": ALL_V2_ENVS['door-close-v2-goal-observable'],
        "seed_dataset": "databoost/envs/metaworld/data/seed/door-close",
        "expert_policy": SawyerDoorCloseV2Policy,
    }),
    "door-lock": AttrDict({
        "env": ALL_V2_ENVS['door-lock-v2-goal-observable'],
        "seed_dataset": "databoost/envs/metaworld/data/seed/door-lock",
        "expert_policy": SawyerDoorLockV2Policy,
    }),
    "door-unlock": AttrDict({
        "env": ALL_V2_ENVS['door-unlock-v2-goal-observable'],
        "seed_dataset": "databoost/envs/metaworld/data/seed/door-unlock",
        "expert_policy": SawyerDoorUnlockV2Policy,
    })
}


'''Seed tasks configs'''
seed_dataset_dir = os.path.join(env_root, "data/seed")
seed_action_noise_pct = 0.1
seed_imgs_res = (640, 480)
num_seed_demos_per_task = 10
seed_tasks_list = [
    "door-open",
    "door-close",
    "door-lock",
    "door-unlock"
]
