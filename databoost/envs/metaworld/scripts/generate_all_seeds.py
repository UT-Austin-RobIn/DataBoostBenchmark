from databoost.envs.metaworld.scripts.generate_dataset import DatasetGeneratorMetaworld
import databoost.envs.metaworld.config as cfg
import os

if __name__=='__main__':
    for task in list(cfg.tasks.keys()):
        seed_dataset_generator = DatasetGeneratorMetaworld(
            **cfg.seed_dataset_kwargs)
        seed_dataset_generator.generate_dataset(
            tasks={task: cfg.tasks[task]},
            dest_dir=os.path.join(cfg.env_root, 'all_seeds'),
            n_demos_per_task=5,#cfg.seed_n_demos,
            do_render=cfg.seed_do_render,
            save_env_and_goal=cfg.seed_save_env_and_goal,
            mask_reward=False
        )