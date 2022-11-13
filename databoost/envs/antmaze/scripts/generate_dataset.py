import copy

import cv2
import numpy as np

from databoost.base import DatasetGenerationPolicyBase, DatasetGeneratorBase
import databoost.envs.antmaze.config as cfg


class DatasetGenerationPolicyAntMaze(DatasetGenerationPolicyBase):
    def __init__(self, antmaze_policy, **datagen_kwargs):
        super().__init__(**datagen_kwargs)
        self.policy = metaworld_policy

    def get_action(self, env, ob: np.ndarray):
        act_noise_pct = self.datagen_kwargs.get("act_noise_pct")
        if act_noise_pct is None:
            act_noise_pct = np.zeros_like(env.action_space.sample())
        act = self.policy.get_action(ob)
        act = np.random.normal(
            act, act_noise_pct * self.datagen_kwargs.act_space_ptp)
        return act


class DatasetGeneratorMetaworld(DatasetGeneratorBase):
    def init_env(self, task_config):
        return initialize_env(task_config.env())

    def init_policy(self, env, task_config):
        act_space = env.action_space
        act_space_ptp = act_space.high - act_space.low
        datagen_kwargs = copy.deepcopy(self.dataset_kwargs)
        datagen_kwargs.update({"act_space_ptp": act_space_ptp})
        return DatasetGenerationPolicyMetaworld(
            task_config.expert_policy,
            **datagen_kwargs
        )

    def get_max_traj_len(self, env, task_config):
        return task_config.env.max_path_length

    def render_img(self, env):
        camera = self.dataset_kwargs.camera
        im = env.render(offscreen=True,
                        camera_name=camera,
                        resolution=self.dataset_kwargs.resolution)[:, :, ::-1]
        if camera == "behindGripper":  # this view requires a 180 rotation
            im = cv2.rotate(im, cv2.ROTATE_180)
        return im

    def is_success(self, env, ob, rew, done, info):
        return info["success"]

    def post_process_step(self, env, ob, rew, done, info):
        info.update({
            "fps": env.metadata['video.frames_per_second'],
            "resolution": self.dataset_kwargs.resolution,
            "act_noise_pct": self.dataset_kwargs.act_noise_pct
        })
        if info['success']: done = True
        return ob, rew, done, info


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
        render = cfg.seed_render,
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
        render = cfg.prior_render,
        mask_reward = True
    )
