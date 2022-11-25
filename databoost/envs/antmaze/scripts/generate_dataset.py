import copy
from typing import Dict, Tuple

import cv2
import gym
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

from databoost.base import \
    DataBoostEnvWrapper, DatasetGenerationPolicyBase, DatasetGeneratorBase
from databoost.utils.general import AttrDict
from databoost.envs.antmaze import DataBoostBenchmarkAntMaze
import databoost.envs.antmaze.config as cfg
from databoost.envs.antmaze.utils import initialize_env


class DatasetGenerationPolicyAntMaze(DatasetGenerationPolicyBase):
    def __init__(self,
                 antmaze_env: gym.Env,
                 antmaze_policy: nn.Module,
                 **datagen_kwargs: AttrDict):
        '''DatasetGenerationPolicyAntMaze is the expert policy
        for generating the offline dataset. It implements a get_action
        function that returns actions given observations.

        Args:
            antmaze_env [gym.Env]: The AntMaze gym environment
            antmaze_policy [nn.Module]: An expert policy that generates actions
                                        given a goal waypoint. Needs to be wrapped
                                        in a path planner that generate waypoints
            datagen_kwargs [AttrDict]: data generation configs; must contain
                                   act_space_ptp (action space point-to-point,
                                   which is the act_space.high - act_space.low
                                   of the environment). Optionally, can contain
                                   act_noise_pct (action noise percentage;
                                   action noise as a percentage of the
                                   act_space_ptp)
        '''
        super().__init__(**datagen_kwargs)
        def _goal_reaching_policy_fn(obs: np.ndarray, goal: Tuple[float]) ->
            Tuple[np.ndarray, Tuple[float]]:
            '''helper function that returns action and next waypoint. Compatible
            with D4RL's AntMaze create_navigation_policy function.
            '''
            goal_x, goal_y = goal
            obs_new = obs[2:-2]
            goal_tuple = np.array([goal_x, goal_y])

            # normalize the norm of the relative goals to in-distribution values
            goal_tuple = goal_tuple / np.linalg.norm(goal_tuple) * 10.0

            new_obs = np.concatenate([obs_new, goal_tuple], -1)
            return antmaze_policy.get_action(new_obs)[0], \
                   (goal_tuple[0] + obs[0], goal_tuple[1] + obs[1])
        self.policy = antmaze_env.create_navigation_policy(_goal_reaching_policy_fn)

        act_noise_pct = self.datagen_kwargs.get("act_noise_pct")
        if act_noise_pct is None:
            # if no noise specified, use 0 noise vector
            act_noise_pct = np.zeros_like(env.action_space.sample())
        self.act_noise = act_noise_pct * self.datagen_kwargs.act_space_ptp

    def get_action(self, ob: np.ndarray) -> np.ndarray:
        '''returns an action given an observation; expert policy.

        Args:
            ob [np.ndarray]: observation
        Returns:
            act [np.ndarray]: corresponding action
        '''
        with torch.no_grad():
            act, _ = self.policy(ob)
            act = np.random.normal(act, self.act_noise)
            act = np.clip(act, -1.0, 1.0)
        return act


class DatasetGeneratorAntMaze(DatasetGeneratorBase):
    def init_env(self, task_config: AttrDict) -> DataBoostEnvWrapper:
        '''creates an AntMaze environment according to the task specification
        and returns the initialized environment to be used for data collection.

        Args:
            task_config [AttrDict]: contains configs for dataset generation;
                                    importantly, contains task_name, expert_policy
                                    for data collection,and any expert_policy_kwargs.
        Returns:
            ant_maze_env [DataBoostEnvWrapper]: the requested AntMaze environment
        '''
        ant_maze_env = DataBoostBenchmarkAntMaze(
                        eval_mode=False).get_env(task_config.task_name)
        return ant_maze_env

    def init_policy(self, env: gym.Env, task_config: AttrDict) -> 
        DatasetGenerationPolicyAntMaze:
        '''initializes the data collection policy according to the task configs.

        Args:
            env [gym.Env]: the AntMaze environment; that was returned by init_env
            task_config [AttrDict]: contains configs for dataset generation;
                                    importantly, contains task_name, expert_policy
                                    for data collection,and any expert_policy_kwargs.
        Returns:
            antmaze_dataset_generation_policy [DatasetGenerationPolicyAntMaze]:
                dataset generation object that implements an "act = get_action(ob)"
                function.
        '''
        # get action space range
        act_space = env.action_space
        act_space_ptp = act_space.high - act_space.low
        datagen_kwargs = copy.deepcopy(self.dataset_kwargs)
        datagen_kwargs.update({"act_space_ptp": act_space_ptp})

        # load the expert policy
        expert_policy_kwargs = task_config.expert_policy_kwargs
        policy_state_dict_path = expert_policy_kwargs.pop("state_dict_path")
        policy = task_config.expert_policy(**expert_policy_kwargs)
        policy.load_state_dict(torch.load(policy_state_dict_path))
        policy.eval()

        # create standaridized data generation policy that implements get_action()
        antmaze_dataset_generation_policy = DatasetGenerationPolicyAntMaze(
                                                env,
                                                policy,
                                                **datagen_kwargs
                                            )
        return antmaze_dataset_generation_policy

    def get_max_traj_len(self, env: gym.Env, task_config: AttrDict) -> int:
        '''returns a maximum trajectory length value that will be used for data
        collection. It is env & task specific

        Args:
            env [gym.Env]: AntMaze gym environment
            task_config [AttrDict]: contains configs for dataset generation;
                                    importantly, contains task_name, expert_policy
                                    for data collection,and any expert_policy_kwargs.
        Returns:
            max_traj_len [int]: the maximum allowed trajectory length of rollouts
        '''
        return self.dataset_kwargs.max_traj_len

    def render_img(self, env: gym.Env) -> np.ndarray:
        '''generates images of the environment

        Args:
            env [gym.Env]: AntMaze gym environment
        Returns:
            im [np.ndarray]: an image representation of the env's current state
        '''
        width, height = self.dataset_kwargs.resolution
        im = env.physics.render(width=width, height=height, depth=False)
        im = cv2.rotate(im[:, :, ::-1], cv2.ROTATE_180)
        return im

    def is_success(self,
                   env: gym.Env,
                   ob: np.ndarray,
                   rew: float,
                   done: bool,
                   info: Dict) -> bool:
        '''evaluates whether the given environment step constitutes a success
        in terms of the task at hand.

        Args:
            env [gym.Env]: gym environment
            ob [np.ndarray]: an observation of the environment this step
            rew [float]: reward received for this env step
            done [bool]: whether the trajectory has reached an end
            info [Dict]: metadata of the environment step
        Returns:
            success [bool]: success flag
        '''
        robot = ob[:2]
        dist = np.linalg.norm(robot - env.target_goal)
        success = dist < self.dataset_kwargs.dist_thresh
        return success

    def post_process_step(self,
                          env: gym.Env,
                          ob: np.ndarray,
                          rew: float,
                          done: bool,
                          info: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        '''post processing step to be executed after the environment step but
        before packing the step's results into the offline dataset h5 file.

        Args:
            env [gym.Env]: gym environment
            ob [np.ndarray]: an observation of the environment this step
            rew [float]: reward received for this env step
            done [bool]: whether the trajectory has reached an end
            info [Dict]: metadata of the environment step
        Returns:
            post-processed env step results [Tuple[np.ndarray, float, bool, Dict]]
        '''
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
            # subset of tasks designated as seed tasks
            task_name: task_config for task_name, task_config in cfg.tasks.items()
            if task_name in cfg.seed_tasks_list
        },
        dest_dir = cfg.seed_dataset_dir,  # where to save the seed dataset
        n_demos_per_task = cfg.seed_n_demos,  # num of demos per seed task
        do_render = cfg.seed_do_render,  # whether images should be rendered
        mask_reward = False
    )

    '''generate prior dataset'''
    prior_dataset_generator = DatasetGeneratorAntMaze(**cfg.prior_dataset_kwargs)
    prior_dataset_generator.generate_dataset(
        # subset of tasks to be included in the unstructured large dataset
        tasks = {
            task_name: task_config for task_name, task_config in cfg.tasks.items()
            if task_name in cfg.prior_tasks_list
        },
        dest_dir = cfg.prior_dataset_dir,  # where to save the prior dataset
        n_demos_per_task = cfg.prior_n_demos,  # num of demos per prior task
        do_render = cfg.prior_do_render,  # whether images should be rendered
        mask_reward = True  # mask rewards (set to 0) to enforce unstructuredness
                            # of the offline dataset)
    )
