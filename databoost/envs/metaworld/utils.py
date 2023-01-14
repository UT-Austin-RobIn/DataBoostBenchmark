import cv2

from metaworld.envs.mujoco.mujoco_env import MujocoEnv

from databoost.base import DataBoostEnvWrapper


def initialize_env(env: MujocoEnv) -> MujocoEnv:
    '''Sets environment attributes to prepare it for
    online use.
    Args:
        env [MujocoEnv]: meta-world Mujoco env object
    Returns:
        env [MujocoEnv]: env with attributes prepared for online use
    '''
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.reset()
    return env


def render(env, **kwargs):
    camera = kwargs["camera"] if "camera" in kwargs else "corner"
    resolution = kwargs["resolution"] if "resolution" in kwargs else (224, 224)
    if isinstance(env, DataBoostEnvWrapper): env = env.env
    im = env.render(
        camera_name=camera,
        resolution=resolution,
        offscreen=True)[:, :, ::-1]
    if camera == "behindGripper":  # this view requires a 180 rotation
        im = cv2.rotate(im, cv2.ROTATE_180)
    return im


def get_env_state(env):
    '''Meta-World state is tuple of (state, goal)
    '''
    return (env.get_env_state(), env.goal)


def load_env_state(env, state):
    '''Load state based on state obtained using get_env_state()
    '''
    # env.set_goal(state[-1])
    # env.reset()
    env.goal = state[-1]
    env.reset()
    env.set_env_state(state[0])
    return env