import cv2
import gym


def initialize_env(env: gym.Env) -> gym.Env:
    '''Initializes the env to be ready for task-specific rollouts; sets the
    target goal to the specified target cell (as a scaled position)

    Args:
        env [gym.Env]: AntMaze env
    Returns:
        env [gym.Env]: initialized AntMaze env
    '''
    env.set_target()
    return env


def render(env, **kwargs):
    width, height = kwargs["resolution"] if "resolution" in kwargs else (224, 224)
    im = env.physics.render(width=width, height=height, depth=False)
    im = cv2.rotate(im[:, :, ::-1], cv2.ROTATE_180)
    return im