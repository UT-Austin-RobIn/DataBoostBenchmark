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
from PIL import Image


class DataBoostBenchmarkLanguageTable(DataBoostBenchmarkBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.r3m = load_r3m("resnet50")  # resnet18, resnet34
        self.r3m.eval()
        self.r3m.to(self.device)
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        n_px = self.clip_model.input_resolution.item()
        self.clip_preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(n_px, interpolation=Image.BICUBIC),
                torchvision.transforms.CenterCrop(n_px),
                # lambda image: image.convert("RGB"),
                # torchvision.transforms.ToTensor()
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        self.act_norm_mean = torch.tensor([0.00011653, -0.0001171], requires_grad = False)
        self.act_norm_stddev = torch.tensor([0.01216072, 0.01504988], requires_grad = False)

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
        # def language_table_preproc(action):
        #     '''unnormalize actions'''
        #     return action * self.act_norm_stddev + self.act_norm_mean

        def language_table_postproc(obs, reward, done, info):
            '''Prepare observation.'''
            if isinstance(obs, dict):
                
                img = torch.from_numpy(obs['rgb'].transpose(2, 0, 1)).to(self.device)[None]

                clip_tokenized_imgs = self.clip_preprocess(torch.tensor(img)/255)
                clip_encs = self.clip_model.encode_image(clip_tokenized_imgs.to(self.device)).data.cpu().numpy()[0]

                # encode images with R3M
                img = torchvision.transforms.functional.crop(img, top=0, left=100, height=img.shape[2], width=img.shape[3]-200)
                img = torchvision.transforms.Resize((224, 224))(img)
                r3m_encs = self.r3m(img).data.cpu().numpy()[0]  # [2048,]
                img_obs = np.concatenate((clip_encs, r3m_encs), axis=-1)

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
                render_func=lambda x: x.render()[:, :, ::-1],
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
