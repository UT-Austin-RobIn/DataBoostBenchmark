from metaworld.envs.mujoco.mujoco_env import MujocoEnv


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
    return env
