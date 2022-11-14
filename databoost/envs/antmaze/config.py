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
    'non_zero_reset': True,
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
            "target_cell": (6, 6)
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
            "target_cell": (1, 6)
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
            "target_cell": (6, 1)
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
            "target_cell": (1, 10)
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
            "target_cell": (7, 10)
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
            "target_cell": (7, 1)
        },
        "seed_dataset": "",
        "expert_policy": TanhGaussianPolicy,
        "expert_policy_kwargs": copy.deepcopy(common_expert_policy_kwargs)
    })
}


'''Prior tasks configs'''
prior_tasks_list = [
    "medium-goal-top-right",
    "medium-goal-bottom-right",
    "medium-goal-bottom-left",
    "large-goal-top-right",
    "large-goal-bottom-right",
    "large-goal-bottom-left"
]
prior_dataset_dir = os.path.join(env_root, "data/prior")
prior_n_demos = 20
prior_do_render = True
prior_action_noise_pct = 0.1
prior_imgs_res = (224, 224)
prior_dataset_kwargs = AttrDict({
    "max_traj_len": 500,
    "dist_thresh": 1.0,
    "act_noise_pct": 0.05,
    "resolution": (224, 224)
})


'''Seed tasks configs'''
seed_tasks_list = [
    "large-goal-bottom-right"
]
seed_dataset_dir = os.path.join(env_root, "data/seed")
seed_n_demos = 10
seed_do_render = True
seed_dataset_kwargs = AttrDict({
    "max_traj_len": 500,
    "dist_thresh": 0.5,
    "act_noise_pct": 0.05,
    "resolution": (224, 224)
})
