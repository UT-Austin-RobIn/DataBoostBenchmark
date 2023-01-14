import copy

import cv2
import numpy as np
import torch

from databoost.base import \
    DataBoostEnvWrapper, DatasetGenerationPolicyBase, DatasetGeneratorBase
from databoost.envs.metaworld import DataBoostBenchmarkMetaworld
import databoost.envs.metaworld.config as cfg
from databoost.envs.metaworld.utils import render, get_env_state


class DatasetGenerationPolicyMetaworld(DatasetGenerationPolicyBase):
    def __init__(self, metaworld_policy, **datagen_kwargs):
        super().__init__(**datagen_kwargs)
        act_noise_pct = self.datagen_kwargs.get("act_noise_pct")
        if act_noise_pct is None:
            act_noise_pct = np.zeros_like(env.action_space.sample())
        self.act_noise = act_noise_pct * self.datagen_kwargs.act_space_ptp
        self.policy = metaworld_policy

    def get_action(self, ob: np.ndarray):
        with torch.no_grad():
            act = self.policy.get_action(ob)
            act = np.random.normal(act, self.act_noise)
            return act


class DatasetGeneratorMetaworld(DatasetGeneratorBase):
    def init_env(self, task_config) -> DataBoostEnvWrapper:
        '''creates an Meta-WOrld environment according to the task specification
        and returns the initialized environment to be used for data collection.

        Args:
            task_config [AttrDict]: contains configs for dataset generation;
                                    importantly, contains task_name, expert_policy
                                    for data collection,and any expert_policy_kwargs.
        Returns:
            env [DataBoostEnvWrapper]: the requested Meta-World environment
        '''
        return DataBoostBenchmarkMetaworld(mask_goal_pos=False).get_env(task_config.task_name)

    def init_policy(self, env, task_config):
        act_space = env.action_space
        act_space_ptp = act_space.high - act_space.low
        datagen_kwargs = copy.deepcopy(self.dataset_kwargs)
        datagen_kwargs.update({"act_space_ptp": act_space_ptp})
        return DatasetGenerationPolicyMetaworld(
            task_config.expert_policy(),
            **datagen_kwargs
        )

    def get_max_traj_len(self, env, task_config):
        return env.max_path_length

    def render_img(self, env):
        im = render(env, **self.dataset_kwargs)
        return im

    def is_success(self, env, ob, rew, done, info):
        return info["success"]

    def post_process_step(self, env, ob, rew, done, info):
        info.update({
            "fps": env.metadata['video.frames_per_second'],
            "resolution": self.dataset_kwargs.resolution
        })
        done = info['success']
        return ob, rew, done, info
    
    def get_env_state(self, env):
        return get_env_state(env)


if __name__ == "__main__":
    '''generate seed dataset'''
    seed_dataset_generator = DatasetGeneratorMetaworld(**cfg.seed_dataset_kwargs)
    seed_dataset_generator.generate_dataset(
        tasks = {
            task_name: task_config for task_name, task_config in cfg.tasks.items()
            if task_name in cfg.seed_tasks_list
        },
        dest_dir = cfg.seed_dataset_dir,
        n_demos_per_task = cfg.seed_n_demos,
        do_render = cfg.seed_do_render,
        save_env_and_goal = cfg.seed_save_env_and_goal,
        mask_reward = False
    )

    '''generate prior dataset'''
    prior_dataset_generator = DatasetGeneratorMetaworld(**cfg.prior_dataset_kwargs)
    prior_dataset_generator.generate_dataset(
        tasks = {
            task_name: task_config for task_name, task_config in cfg.tasks.items()
            if task_name in cfg.prior_tasks_list
        },
        dest_dir = cfg.prior_dataset_dir,
        n_demos_per_task = cfg.prior_n_demos,
        do_render = cfg.prior_do_render,
        save_env_and_goal = cfg.prior_save_env_and_goal,
        mask_reward = True
    )

    '''generate test dataset'''
    test_dataset_generator = DatasetGeneratorMetaworld(**cfg.test_dataset_kwargs)
    test_dataset_generator.generate_dataset(
        tasks = {
            task_name: task_config for task_name, task_config in cfg.tasks.items()
            if task_name in cfg.test_tasks_list
        },
        dest_dir = cfg.test_dataset_dir,
        n_demos_per_task = cfg.test_n_demos,
        do_render = cfg.test_do_render,
        save_env_and_goal = cfg.test_save_env_and_goal,
        mask_reward = False
    )
