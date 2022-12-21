from collections import defaultdict as dd
import os
import random

from calvin_env.envs.play_table_env import PlayTableSimEnv
from gym import spaces
import hydra
import numpy as np

from databoost.base import DataBoostEnvWrapper, DataBoostBenchmarkBase
from databoost.utils.general import AttrDict
import databoost.envs.metaworld.config as cfg


class CalvinEnv(PlayTableSimEnv):
    def __init__(self, task, scene='D'):
        self.ownsPhysicsClient = False

        # Read from calvin configs
        with hydra.initialize(config_path="calvin_env_conf"):
            cfg = hydra.compose(
                config_name="config_data_collection.yaml",
                overrides=["cameras=static_and_gripper", f"scene=calvin_scene_{scene}"],
            )
            cfg.env["use_egl"] = False
            cfg.env["show_gui"] = False
            cfg.env["use_vr"] = False
            cfg.env["use_scene_info"] = True

        new_env_cfg = {**cfg.env}
        new_env_cfg.pop("_target_", None)
        new_env_cfg.pop("_recursive_", None)
        PlayTableSimEnv.__init__(self, **new_env_cfg)

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(39,))
        self.tasks = hydra.utils.instantiate(cfg.tasks)
        self.task = task

    def reset(self):
        obs = super().reset()
        self.start_info = self.get_info()
        return obs

    def get_obs(self):
        robot_obs, robot_info = self.robot.get_observation()
        state_obs = self.get_state_obs()
        obs = np.concatenate([robot_obs, state_obs["scene_obs"]])
        return obs

    def _success(self):
        """ Returns a boolean indicating if the task was performed correctly """
        current_info = self.get_info()
        task_filter = [self.task]
        task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, task_filter)
        return self.task in task_info

    def _reward(self):
        """ Returns the reward function that will be used 
        for the RL algorithm """
        reward = int(self._success())
        r_info = {'reward': reward}
        return reward, r_info

    def _termination(self):
        """ Indicates if the robot has reached a terminal state """
        success = self._success()
        done = success
        d_info = {'success': success}        
        return done, d_info

    def step(self, action):
            """ Performing a relative action in the environment
                input:
                    action: 7 tuple containing
                            Position x, y, z. 
                            Angle in rad x, y, z. 
                            Gripper action
                            each value in range (-1, 1)

                            OR
                            8 tuple containing
                            Relative Joint angles j1 - j7 (in rad)
                            Gripper action
                output:
                    observation, reward, done info
            """
            # Transform gripper action to discrete space
            env_action = action.copy()
            env_action[-1] = (int(action[-1] >= 0) * 2) - 1

            # for using actions in joint space
            if len(env_action) == 8:
                env_action = {"action": env_action, "type": "joint_rel"}

            self.robot.apply_action(env_action)
            for i in range(self.action_repeat):
                self.p.stepSimulation(physicsClientId=self.cid)
            obs = self.get_obs()
            info = self.get_info()
            reward, r_info = self._reward()
            done, d_info = self._termination()
            info.update(r_info)
            info.update(d_info)
            return obs, reward, done, info

    def render(self, mode="rgb_array"):
        return super().render(mode=mode)

    def reset_to_state(self, state):
        return self.reset(robot_obs=state[:15], scene_obs=state[15:])
