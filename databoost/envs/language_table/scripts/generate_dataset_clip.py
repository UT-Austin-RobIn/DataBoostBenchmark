import tqdm
import torch
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Dict
from databoost.utils.general import AttrDict
from databoost.utils.data import write_h5
import clip
from torchvision.transforms import ToPILImage


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
        self._dataset_directories = {
            'language_table': 'gs://gresearch/robotics/language_table',
            'language_table_sim': 'gs://gresearch/robotics/language_table_sim',
            'language_table_blocktoblock_sim': 'gs://gresearch/robotics/language_table_blocktoblock_sim',
            'language_table_blocktoblock_4block_sim': 'gs://gresearch/robotics/language_table_blocktoblock_4block_sim',
            'language_table_blocktoblock_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoblock_oracle_sim',
            'language_table_blocktoblockrelative_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoblockrelative_oracle_sim',
            'language_table_blocktoabsolute_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim',
            'language_table_blocktorelative_oracle_sim': 'gs://gresearch/robotics/language_table_blocktorelative_oracle_sim',
            'language_table_separate_oracle_sim': 'gs://gresearch/robotics/language_table_separate_oracle_sim',
        }
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to_pil_transform = ToPILImage()
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', self.device)
        self.clip_model.eval()
        self.clip_model.to(self.device)

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

    def generate_dataset(self,
                         dataset_name: str,
                         dest_dir: str,
                         n_episodes: int = -1,
                         mask_reward: bool = True):
        '''generates a dataset given a list of tasks and other configs.
        Args:
            dataset_name [str]: Name of the dataset to be converted
            dest_dir [str]: path to directory to which the dataset is to be written
            n_episodes [int]: number of episodes to convert to HDF5, default: convert all in train split
            mask_reward [bool]: if true, all rewards are set to zero (for prior dataset)
        '''
        # make dataset
        dataset_path = os.path.join(self._dataset_directories[dataset_name], '0.0.1')
        builder = tfds.builder_from_directory(dataset_path)
        episode_ds = builder.as_dataset(split='train[:50%]')

        # choose data size
        if n_episodes < 0:
            n_total = tf.data.experimental.cardinality(episode_ds)
            it = episode_ds
        else:
            n_total = n_episodes
            it = episode_ds.take(n_episodes)
        print(f"Converting {n_total} episodes from {dataset_name}!")

        # make data dir
        os.makedirs(dest_dir, exist_ok=True)

        # iterate over data and save to H5
        for i, episode in tqdm.tqdm(enumerate(it)):
            traj = self.init_traj()
            for step in episode['steps']:
                ob = tf.concat((step['observation']['effector_translation'],
                                step['observation']['effector_target_translation']), axis=0)
                act = step['action']
                rew = step['reward']
                done = step['is_last'] or step['is_terminal']
                info = AttrDict(instruction=step['observation']['instruction'])
                im = np.transpose(step['observation']['rgb'], (2, 0, 1))
                im = self.clip_preprocess(self.to_pil_transform(im)).numpy()
                if mask_reward: rew = 0.0

                self.add_to_traj(traj, im, act, rew, done, info)

            # move trajectory data to numpy
            traj = self.traj_to_numpy(traj)
            traj.observations = self.clip_model.encode_image(torch.from_numpy(traj.observations).to(self.device)).data.cpu().numpy()

            filename = f"episode_{i}"
            traj["dones"][-1] = True
            write_h5(traj, os.path.join(dest_dir, filename + ".h5"))


if __name__ == "__main__":
    dest_dir = "data/lang"
    DatasetSaver().generate_dataset(
        dataset_name='language_table_sim',
        dest_dir='data_and_models/language_table/data',
        #n_episodes=1000,
    )
