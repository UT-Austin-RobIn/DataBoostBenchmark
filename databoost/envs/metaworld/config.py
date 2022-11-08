import os

from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerDoorEnvV2,
    SawyerDoorCloseEnvV2,
    SawyerDoorLockEnvV2,
    SawyerDoorUnlockEnvV2,
)

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
        "env": SawyerDoorEnvV2,
        "expert_policy": SawyerDoorOpenV2Policy,
    },
    "door-close": {
        "env": SawyerDoorCloseEnvV2,
        "expert_policy": SawyerDoorCloseV2Policy,
    },
    "door-lock": {
        "env": SawyerDoorLockEnvV2,
        "expert_policy": SawyerDoorLockV2Policy,
    },
    "door-unlock": {
        "env": SawyerDoorUnlockEnvV2,
        "expert_policy": SawyerDoorUnlockV2Policy,
    }
}
