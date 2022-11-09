import argparse
import pickle

import cv2

from databoost.utils.data import read_h5


def visualize_h5(path: str):
    dest_path = path.replace(".h5", ".avi")
    traj_data = read_h5(path)
    traj_metadata = pickle.loads(traj_data.infos[0])
    fps = traj_metadata["fps"]
    res = traj_metadata["resolution"]
    writer = cv2.VideoWriter(
        dest_path,
        cv2.VideoWriter_fourcc('M','J','P','G'),
        fps,
        res
    )
    for im in traj_data.imgs:
        writer.write(im)
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize trajectory of h5 file.')
    parser.add_argument("--path", help="path to h5 file")
    args = parser.parse_args()
    visualize_h5(args.path)
