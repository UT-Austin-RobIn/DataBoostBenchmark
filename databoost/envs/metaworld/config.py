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
        "task_name": "door-open",
        "env": ALL_V2_ENVS['door-open-v2-goal-observable'],
        "seed_dataset": os.path.join(env_root, "data/seed/door-open"),
        "expert_policy": SawyerDoorOpenV2Policy,
    }),
    "door-close": AttrDict({
        "task_name": "door-close",
        "env": ALL_V2_ENVS['door-close-v2-goal-observable'],
        "seed_dataset": os.path.join(env_root, "data/seed/door-close"),
        "expert_policy": SawyerDoorCloseV2Policy,
    }),
    "door-lock": AttrDict({
        "task_name": "door-lock",
        "env": ALL_V2_ENVS['door-lock-v2-goal-observable'],
        "seed_dataset": os.path.join(env_root, "data/seed/door-lock"),
        "expert_policy": SawyerDoorLockV2Policy,
    }),
    "door-unlock": AttrDict({
        "task_name": "door-unlock",
        "env": ALL_V2_ENVS['door-unlock-v2-goal-observable'],
        "seed_dataset": os.path.join(env_root, "data/seed/door-unlock"),
        "expert_policy": SawyerDoorUnlockV2Policy,
    })
}


'''Seed tasks configs'''
seed_tasks_list = [
    "door-open",
    "door-close",
    "door-lock",
    "door-unlock"
]
seed_dataset_dir = os.path.join(env_root, "data/seed")
seed_n_demos = 10
seed_do_render = True
seed_dataset_kwargs = AttrDict({
    "act_noise_pct": 0.1,
    "resolution": (224, 224),
    "camera": "corner"
})


'''Prior tasks configs'''
prior_tasks_list = [
    "door-open",
    "door-close",
    "door-lock",
    "door-unlock"
]
prior_dataset_dir = os.path.join(env_root, "data/prior")
prior_n_demos = 30
prior_do_render = True
prior_dataset_kwargs = AttrDict({
    "act_noise_pct": 0.1,
    "resolution": (224, 224),
    "camera": "corner"
})
