import copy

import cv2
from PIL import Image
import numpy as np
import torch

from databoost.base import DatasetGenerationPolicyBase, DatasetGeneratorBase
from databoost.envs.antmaze import DataBoostBenchmarkAntMaze
import databoost.envs.antmaze.config as cfg
from databoost.envs.antmaze.utils import initialize_env


class DatasetGenerationPolicyAntMaze(DatasetGenerationPolicyBase):
    def __init__(self, antmaze_env, antmaze_policy, **datagen_kwargs):
        super().__init__(**datagen_kwargs)
        def _goal_reaching_policy_fn(obs, goal):
            goal_x, goal_y = goal
            obs_new = obs[2:-2]
            goal_tuple = np.array([goal_x, goal_y])

            # normalize the norm of the relative goals to in-distribution values
            goal_tuple = goal_tuple / np.linalg.norm(goal_tuple) * 10.0

            new_obs = np.concatenate([obs_new, goal_tuple], -1)
            return antmaze_policy.get_action(new_obs)[0], (goal_tuple[0] + obs[0], goal_tuple[1] + obs[1])
        self.policy = antmaze_env.create_navigation_policy(_goal_reaching_policy_fn)
        
        act_noise_pct = self.datagen_kwargs.get("act_noise_pct")
        if act_noise_pct is None:
            act_noise_pct = np.zeros_like(env.action_space.sample())
        self.act_noise = act_noise_pct * self.datagen_kwargs.act_space_ptp

    def get_action(self, ob: np.ndarray):
        with torch.no_grad():
            act, _ = self.policy(ob)
            act = np.random.normal(act, self.act_noise)
            act = np.clip(act, -1.0, 1.0)
        return act


class DatasetGeneratorAntMaze(DatasetGeneratorBase):
    def init_env(self, task_config):
        return DataBoostBenchmarkAntMaze(eval_mode=False).get_env(task_config.task_name)

    def init_policy(self, env, task_config):
        act_space = env.action_space
        act_space_ptp = act_space.high - act_space.low
        datagen_kwargs = copy.deepcopy(self.dataset_kwargs)
        datagen_kwargs.update({"act_space_ptp": act_space_ptp})

        expert_policy_kwargs = task_config.expert_policy_kwargs
        policy_state_dict_path = expert_policy_kwargs.pop("state_dict_path")
        policy = task_config.expert_policy(**expert_policy_kwargs)
        policy.load_state_dict(torch.load(policy_state_dict_path))
        policy.eval()

        return DatasetGenerationPolicyAntMaze(
            env,
            policy,
            **datagen_kwargs
        )

    def get_max_traj_len(self, env, task_config):
        return self.dataset_kwargs.max_traj_len

    def render_img(self, env):
        width, height = self.dataset_kwargs.resolution
        im = env.physics.render(width=width, height=height, depth=False)
        im = cv2.rotate(im[:, :, ::-1], cv2.ROTATE_180)
        return im

    def is_success(self, env, ob, rew, done, info):
        robot = ob[:2]
        dist = np.linalg.norm(robot - env.target_goal)
        return dist < self.dataset_kwargs.dist_thresh

    def post_process_step(self, env, ob, rew, done, info):
        info.update({
            "fps": 20,
            "resolution": self.dataset_kwargs.resolution,
            "qpos": env.physics.data.qpos.ravel().copy(),
            "qvel": env.physics.data.qvel.ravel().copy()
        })
        done = self.is_success(env, ob, rew, done, info)
        ob = ob[:-2]  # remove goal direction, same as D4RL
        return ob, rew, done, info


if __name__ == "__main__":
    '''generate seed dataset'''
    seed_dataset_generator = DatasetGeneratorAntMaze(**cfg.seed_dataset_kwargs)
    seed_dataset_generator.generate_dataset(
        tasks = {
            task_name: task_config for task_name, task_config in cfg.tasks.items()
            if task_name in cfg.seed_tasks_list
        },
        dest_dir = cfg.seed_dataset_dir,
        n_demos_per_task = cfg.seed_n_demos,
        do_render = cfg.seed_do_render,
        mask_reward = False
    )

    '''generate prior dataset'''
    prior_dataset_generator = DatasetGeneratorAntMaze(**cfg.prior_dataset_kwargs)
    prior_dataset_generator.generate_dataset(
        tasks = {
            task_name: task_config for task_name, task_config in cfg.tasks.items()
            if task_name in cfg.prior_tasks_list
        },
        dest_dir = cfg.prior_dataset_dir,
        n_demos_per_task = cfg.prior_n_demos,
        do_render = cfg.prior_do_render,
        mask_reward = True
    )
