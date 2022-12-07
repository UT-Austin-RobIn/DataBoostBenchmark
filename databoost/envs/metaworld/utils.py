import cv2

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


def render(env, **kwargs):
    camera = kwargs["camera"] if "camera" in kwargs else "corner"
    resolution = kwargs["resolution"] if "resolution" in kwargs else (224, 224)
    im = env.render(
        camera_name=camera,
        resolution=resolution,
        offscreen=True)[:, :, ::-1]
    if camera == "behindGripper":  # this view requires a 180 rotation
        im = cv2.rotate(im, cv2.ROTATE_180)
    return im
