import argparse
import os
import pickle

import cv2

from databoost.utils.data import read_h5, find_h5


def visualize_h5(path: str):
    '''generate a .avi video based on h5 trajectory. Assumes the h5 file contains
    an imgs attribute. The video will be written to a directory called "movies"
    at the same directory level as the original h5 file.

    Args:
        path [str]: path to file/directory of h5 file(s). If path is a directory,
                    render videos of all h5 files under the directory
    '''
    # if path is directory, visualize all h5 files under directory
    paths = find_h5(path) if os.path.isdir(path) else [path]

    for path in paths:
        # save videos to in a directory called "movies" at the same directory
        # level as the original h5 file
        root, filename = os.path.split(path)
        dest_filename = filename.replace(".h5", ".avi").replace(".hdf5", ".avi")
        dest_root = os.path.join(root, "movies")
        os.makedirs(dest_root, exist_ok=True)
        dest_path = os.path.join(dest_root, dest_filename)
        # load visualization metadata
        traj_data = read_h5(path)
        try:
            fps = traj_data.infos.fps[0]
        except Exception:
            fps = 20
        try:
            resolution = traj_data.infos.resolution[0]
        except Exception:
            resolution = (224, 224)
        # write images of dataset to a video and save
        writer = cv2.VideoWriter(
            dest_path,
            cv2.VideoWriter_fourcc('M','J','P','G'),
            fps,
            resolution
        )
        for im in traj_data.imgs:
            writer.write(im)
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize trajectory of h5 file(s) at path/dir.')
    parser.add_argument("--path", help="path to h5 file/dir")
    args = parser.parse_args()
    visualize_h5(args.path)
