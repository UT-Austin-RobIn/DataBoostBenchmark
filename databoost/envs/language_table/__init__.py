import copy
import os
from typing import Dict, Any

import gym
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from databoost.base import DataBoostEnvWrapper, DataBoostBenchmarkBase
from databoost.utils.general import AttrDict
from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block


class DataBoostBenchmarkLanguageTable(DataBoostBenchmarkBase):

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
        env = language_table.LanguageTable(
            block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
            reward_factory=block2block.BlockToBlockReward,
            seed=0
        )
        env = DataBoostEnvWrapper(
                env,
                seed_dataset_url="/data/karl/data/table_sim/prior_data",
                prior_dataset_url="/data/karl/data/table_sim/prior_data",
                test_dataset_url="/data/karl/data/table_sim/prior_data",
                render_func=lambda x: x.render(),
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


class DataBoostEnvWrapperLanguageTable(DataBoostEnvWrapper):
    '''Overwrites data loading with tfds data pipeline.'''

    def _get_dataset(self, dataset_dir: str, n_demos: int = None) -> AttrDict:
        raise NotImplementedError('Language Table does not support loading of whole dataset at once.')

    def _get_dataloader(self,
                        dataset_dir: str,
                        n_demos: int = None,
                        seq_len: int = None,
                        batch_size: int = 1,
                        shuffle: bool = True,
                        load_imgs: bool = True,
                        goal_condition: bool = False) -> Any: #DataLoader:
        '''gets a dataloader to load in h5 data from the given dataset_dir.

        Args:
            dataset_dir [str]: directory of h5 data
            n_demos [int]: the number of demos (h5 files) to retrieve from the
                           dataset dir
            seq_len [int]: the window length with which to split demonstrations
            batch_size [int]: number of sequences to load in as a batch
            shuffle [bool]: shuffle sequences to be loaded
        Returns:
            dataloader [DataLoader]: DataLoader for given dataset directory
        '''
        def prep_episode(episode):
            """Samples random subsequence, extracts observation, builds output dict."""
            # sample random start step
            #steps = list(iter(episode['steps']))
            #start_step = np.random.randint(0, len(steps) - seq_len - 1)

            step = next(iter(episode['steps']))
            return {'observations': tf.concat((step['observation']['effector_translation'],
                                               step['observation']['effector_target_translation']), axis=-1)[None],
                    'actions': step['action'][None],
            }

            obs, act = [], []
            for step in episode['steps']: #steps[start_step : start_step + seq_len]:
                obs.append(
                    tf.concat((step['observation']['effector_translation'],
                               step['observation']['effector_target_translation']), axis=-1))
                act.append(step["action"])
                if len(obs) == 2: break
            return {
                "observations": obs, #tf.stack(obs, axis=0),
                "actions": act, #tf.stack(act, axis=0),
            }

        builder = tfds.builder_from_directory(os.path.join(dataset_dir, '0.0.1'))
        ds = builder.as_dataset(split='train[:2560]')
        ds = ds.cache()
        #ds = ds.map(prep_episode)

        if shuffle:
            ds = ds.shuffle(1024)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        tfds.benchmark(ds)
        tfds.benchmark(ds)
        return tfds.as_numpy(ds)


__all__ = [DataBoostBenchmarkLanguageTable]
