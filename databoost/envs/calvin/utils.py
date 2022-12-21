import cv2


def render(env, **kwargs):
    resolution = kwargs["resolution"] if "resolution" in kwargs else (224, 224)
    mode = kwargs["mode"] if "mode" in kwargs else "rgb_array"
    im = env.render(
        mode=mode)[:, :, ::-1]
    return cv2.resize(im, resolution)