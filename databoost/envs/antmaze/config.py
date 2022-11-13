import copy
import os

import gym
import d4rl
from d4rl.locomotion import maze_env
from d4rl.locomotion.ant import AntMazeEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy

from databoost.utils.general import AttrDict


'''General configs'''
env_root = "/home/jullian-yapeter/code/DataBoostBenchmark/databoost/envs/antmaze"


'''Tasks configs'''
common_task_kwargs = {
    'reward_type':'sparse',
    'non_zero_reset':True,
    'eval':True,
    'maze_size_scaling': 4.0,
    'ref_min_score': 0.0,
    'ref_max_score': 1.0,
    'v2_resets': True,
}
common_expert_policy_kwargs = {
    "state_dict_path": os.path.join(env_root, "data/policy/antmaze_expert_policy_state_dict.pt"),
    "hidden_sizes": [256, 256],
    "obs_dim": 29,
    "action_dim": 8
}
tasks = {
    "medium-goal-top-right": AttrDict({
        "task_name": "medium-goal-top-right",
        "env": AntMazeEnv,
        "env_kwargs": {
            **copy.deepcopy(common_task_kwargs),
            "maze_map": maze_env.BIG_MAZE,
            "goal_sampler": lambda _: (6, 6)
        },
        "seed_dataset": "",
        "expert_policy": TanhGaussianPolicy,
        "expert_policy_kwargs": copy.deepcopy(common_expert_policy_kwargs)
    }),
    "medium-goal-bottom-right": AttrDict({
        "task_name": "medium-goal-bottom-right",
        "env": AntMazeEnv,
        "env_kwargs": {
            **copy.deepcopy(common_task_kwargs),
            "maze_map": maze_env.BIG_MAZE,
            "goal_sampler": lambda _: (1, 6)
        },
        "seed_dataset": "",
        "expert_policy": TanhGaussianPolicy,
        "expert_policy_kwargs": copy.deepcopy(common_expert_policy_kwargs)
    }),
    "medium-goal-bottom-left": AttrDict({
        "task_name": "medium-goal-bottom-left",
        "env": AntMazeEnv,
        "env_kwargs": {
            **copy.deepcopy(common_task_kwargs),
            "maze_map": maze_env.BIG_MAZE,
            "goal_sampler": lambda _: (6, 1)
        },
        "seed_dataset": "",
        "expert_policy": TanhGaussianPolicy,
        "expert_policy_kwargs": copy.deepcopy(common_expert_policy_kwargs)
    }),
    "large-goal-top-right": AttrDict({
        "task_name": "large-goal-top-right",
        "env": AntMazeEnv,
        "env_kwargs": {
            **copy.deepcopy(common_task_kwargs),
            "maze_map": maze_env.HARDEST_MAZE,
            "goal_sampler": lambda _: (1, 10)
        },
        "seed_dataset": "",
        "expert_policy": TanhGaussianPolicy,
        "expert_policy_kwargs": copy.deepcopy(common_expert_policy_kwargs)
    }),
    "large-goal-bottom-right": AttrDict({
        "task_name": "large-goal-bottom-right",
        "env": AntMazeEnv,
        "env_kwargs": {
            **copy.deepcopy(common_task_kwargs),
            "maze_map": maze_env.HARDEST_MAZE,
            "goal_sampler": lambda _: (7, 10)
        },
        "seed_dataset": "",
        "expert_policy": TanhGaussianPolicy,
        "expert_policy_kwargs": copy.deepcopy(common_expert_policy_kwargs)
    }),
    "large-goal-bottom-left": AttrDict({
        "task_name": "large-goal-bottom-left",
        "env": AntMazeEnv,
        "env_kwargs": {
            **copy.deepcopy(common_task_kwargs),
            "maze_map": maze_env.HARDEST_MAZE,
            "goal_sampler": lambda _: (7, 1)
        },
        "seed_dataset": "",
        "expert_policy": TanhGaussianPolicy,
        "expert_policy_kwargs": copy.deepcopy(common_expert_policy_kwargs)
    })
}


'''Prior tasks configs'''
prior_dataset_dir = os.path.join(env_root, "data/prior")
prior_action_noise_pct = 0.1
prior_imgs_res = (224, 224)
num_prior_demos_per_task = 20
prior_tasks_list = [
    "medium-goal-top-right",
    "medium-goal-bottom-right",
    "medium-goal-bottom-left",
    "large-goal-top-right",
    "large-goal-bottom-right",
    "large-goal-bottom-left"
]


'''Seed tasks configs'''
seed_dataset_dir = os.path.join(env_root, "data/seed")
seed_action_noise_pct = 0.1
seed_imgs_res = (224, 224)
num_seed_demos_per_task = 10
seed_camera = "corner"
seed_tasks_list = [
    "large-goal-bottom-right"
]
