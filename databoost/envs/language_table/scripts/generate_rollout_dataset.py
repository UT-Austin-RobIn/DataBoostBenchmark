import clip
import tqdm
import torch
import torchvision
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Dict
from databoost.utils.general import AttrDict
from databoost.utils.data import write_h5
from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block, separate_blocks
from r3m import load_r3m


class DatasetSaver:
    def __init__(self):
        self.traj_keys = [
            "observations",
            "actions",
            "rewards",
            "dones",
            "infos",
            "imgs"
        ]

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.r3m = load_r3m("resnet50")  # resnet18, resnet34
        self.r3m.eval()
        self.r3m.to(self.device)
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

    def init_traj(self) -> Dict:
        '''Initialize an empty trajectory, preparing for data collection
        Returns:
            traj [Dict]: dict of empty attributes
        '''
        traj = AttrDict()
        for attr in self.traj_keys:
            traj[attr] = [] if attr not in ("info", "infos") else {}
        return traj

    def add_to_traj(self,
                    traj: AttrDict,
                    ob: np.ndarray,
                    act: np.ndarray,
                    rew: float,
                    done: bool,
                    info: Dict,
                    im: np.ndarray = None):
        '''helper function to append a step's results to a trajectory dictionary
        Args:
            traj [AttrDict]: dictionary with keys {
                observations, actions, rewards, dones, infos, imgs}
            ob [np.ndarray]: env-specific observation
            act [np.ndarray]: env-specific action
            rew [float]: reward; float
            done [bool]: done flag
            info [Dict]: task-specific info
            im [np.ndarray]: rendered image after the step
        '''
        traj.observations.append(ob)
        traj.actions.append(act)
        traj.rewards.append(rew)
        traj.dones.append(done)
        for attr in info:
            if attr not in traj.infos:
                traj.infos[attr] = []
            traj.infos[attr].append(info[attr])
        if im is not None:
            traj.imgs.append(im)

    def traj_to_numpy(self, traj: AttrDict) -> AttrDict:
        '''convert trajectories attributes into numpy arrays

        Args:
            traj [AttrDict]: dictionary with keys {obs, acts, rews, dones, infos, ims}
        Returns:
            traj_numpy [AttrDict]: trajectory dict with attributes as numpy arrays
        '''
        traj_numpy = self.init_traj()
        for attr in traj:
            if attr not in ("info", "infos"):
                traj_numpy[attr] = np.array(traj[attr])
            else:
                for info_attr in traj.infos:
                    traj_numpy.infos[info_attr] = np.array(traj.infos[info_attr])
        return traj_numpy

    def _postprocess_obs(self, obs):
        # encode images with R3M
        img = torch.from_numpy(obs['rgb'].transpose(2, 0, 1)).to(self.device)[None]
        img = torchvision.transforms.Resize((224, 224))(img)
        img_obs = self.r3m(img).data.cpu().numpy()[0]  # [2048,]

        # encode text with CLIP
        inst = obs['instruction']
        decoded_instruction = bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")
        text_tokens = clip.tokenize(decoded_instruction)  # .float()[0]  # [77,]
        text_tokens = self.clip_model.encode_text(
            text_tokens.to(self.device)).data.cpu().numpy()[0]  # [512,]
        obs = np.concatenate((img_obs, text_tokens), axis=-1)
        return obs

    def generate_dataset(self,
                         policy_checkpt: str,
                         dest_dir: str,
                         env_seed: int,
                         n_episodes: int = 100000,
                         max_rollout_len=100):
        '''generates a dataset given a list of tasks and other configs.
        Args:
            policy_checkpt [str]: Filepath to the policy checkpoint
            dest_dir [str]: path to directory to which the dataset is to be written
            n_episodes [int]: number of episodes to collect, default: 100000
        '''
        # load policy
        policy = torch.load(policy_checkpt).eval().to(self.device)

        # make environment
        env = language_table.LanguageTable(
            block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
            reward_factory=separate_blocks.SeparateBlocksReward,
            seed=env_seed,
        )

        # make data dir
        os.makedirs(dest_dir, exist_ok=True)

        # collect rollouts
        for i in tqdm.tqdm(range(n_episodes)):
            ob = self._postprocess_obs(env.reset())
            traj = self.init_traj()
            for _ in range(max_rollout_len - 1):
                with torch.no_grad():
                    act = policy.get_action(ob)
                ob_next, rew, done, info = env.step(act)
                ob_next = self._postprocess_obs(ob_next)
                self.add_to_traj(traj, ob, act, rew, done, info)
                if done and rew > 0:
                    break
                ob = ob_next

            # move trajectory data to numpy
            traj = self.traj_to_numpy(traj)

            # write episode to h5 file
            write_h5(traj, os.path.join(dest_dir, f"episode_{i}.h5"))

if __name__ == "__main__":
    BATCH = 1
    DatasetSaver().generate_dataset(
        policy_checkpt='/home/jullian-yapeter/data/DataBoostBenchmark/language_table/models/dummy/separate/BC_Mixed/language_table-separate-BC_Mixed-goal_cond_False-mask_goal_pos_False-best.pt',
        dest_dir=f'/data/karl/data/table_sim/rollout_data/batch{BATCH}',
        env_seed=BATCH,
        #n_episodes=1000,
    )
