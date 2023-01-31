import clip
import copy
import os
from typing import Dict, Any

import gym
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torchvision

from databoost.base import DataBoostEnvWrapper, DataBoostBenchmarkBase
from databoost.utils.general import AttrDict
from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block, separate_blocks
from r3m import load_r3m


class DataBoostBenchmarkLanguageTable(DataBoostBenchmarkBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.r3m = load_r3m("resnet50")  # resnet18, resnet34
        self.r3m.eval()
        self.r3m.to(self.device)
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

    def get_env(self, task_name: str) -> DataBoostEnvWrapper:
        '''get the wrapped gym environment corresponding to the specified task.

        Args:
            task_name [str]: the name of the task; must be from the list of
                             tasks compatible with this benchmark (self.tasks_list)
        Returns:
            env [DataBoostEnvWrapper]: wrapped env that implements getters for
                                       the corresponding seed and prior offline
                                       datasets
        '''
        def language_table_postproc(obs, reward, done, info):
            '''Prepare observation.'''
            if isinstance(obs, dict):
                # encode images with R3M
                img = torch.from_numpy(obs['rgb'].transpose(2, 0, 1)).to(self.device)[None]
                img = torchvision.transforms.Resize((224, 224))(img)
                img_obs = self.r3m(img).data.cpu().numpy()[0]  # [2048,]

                # encode text with CLIP
                inst = obs['instruction']
                decoded_instruction = bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")
                text_tokens = clip.tokenize(decoded_instruction)#.float()[0]  # [77,]
                text_tokens = self.clip_model.encode_text(
                        text_tokens.to(self.device)).data.cpu().numpy()[0]   # [512,]
                obs = np.concatenate((img_obs, text_tokens), axis=-1)

            return obs, reward, done, info

        env = language_table.LanguageTable(
            block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
            reward_factory=separate_blocks.SeparateBlocksReward,
            seed=0
        )
        env = DataBoostEnvWrapper(
                env,
                seed_dataset_url="/data/karl/data/table_sim/prior_data",
                prior_dataset_url="/data/karl/data/table_sim/prior_data",
                test_dataset_url="/data/karl/data/table_sim/prior_data",
                render_func=lambda x: x.render(),
                postproc_func=language_table_postproc,
            )
        return env

    def evaluate_success(self,
                         env: gym.Env,
                         ob: np.ndarray,
                         rew: float,
                         done: bool,
                         info: Dict) -> bool:
        '''evaluates whether the given environment step constitutes a success
        in terms of the task at hand. This is used in the benchmark's policy
        evaluator.

        Args:
            env [gym.Env]: gym environment
            ob [np.ndarray]: an observation of the environment this step
            rew [float]: reward received for this env step
            done [bool]: whether the trajectory has reached an end
            info [Dict]: metadata of the environment step
        Returns:
            success [bool]: success flag
        '''
        return done and rew > 0


__all__ = [DataBoostBenchmarkLanguageTable]
