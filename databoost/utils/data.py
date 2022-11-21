import json
import os
from typing import Any, List, Dict, Tuple

import h5py
import numpy as np

from databoost.utils.general import AttrDict


def find_h5(dir: str) -> List[str]:
    '''Get list of h5 file paths
    Args:
        dir [str]: path to directory of interest
    Returns:
        file_paths [List[str]]: h5 file paths within dir
    '''
    file_paths = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith((".h5", ".hdf5")):
                file_paths.append(os.path.join(root, file))
    file_paths.sort()
    return file_paths


def read_h5(path: str) -> Dict:
    '''Read h5 file
    Args:
        path [str]: path to the h5 file
    Returns:
        data [Dict]: data of h5 file
    '''
    with h5py.File(path) as F:
        data = AttrDict()
        for key in F.keys():
            if key == "infos":
                data[key] = AttrDict()
                for sub_key in F[key].keys():
                    data[key][sub_key] = F[f"{key}/{sub_key}"][()]
            else:
                data[key] = F[key][()]
    return data


def write_h5(data: Dict, dest_path: str):
    '''Copy selected trajectory of a dataset to the destination directory as an h5 file.
    Args:
        data [AttrDict]: dictionary of data to be written to an h5 file
        dest_path [str]: path of destination file
    '''
    with h5py.File(dest_path, "w") as F:
        for attr, val in data.items():
            if isinstance(val, dict):
                for sub_attr in val:
                    F[f"{attr}/{sub_attr}"] = val[sub_attr]
            else:
                F[attr] = val


def read_json(path: str) -> Any:
    '''Read JSON file
    Args:
        path [str]: path to the JSON file
    Returns:
        data [Any]: data of JSON file
    '''
    with open(path, "r") as F:
        return json.load(F)


def write_json(obj: Any, dest_path: str):
    '''Write JSON file to destination path
    Args:
        obj [Any]: data to be written to JSON
        dest_path [str]: destination path of the JSON file
    '''
    with open(dest_path, "w") as F:
        json.dump(obj, F)


def concatenate_traj_data(trajs: Tuple[AttrDict]):
    traj_concat = AttrDict()
    for traj in trajs:
        traj_len = len(traj.observations)
        for attr, val in traj.items():
            assert isinstance(traj[attr], np.ndarray)
            assert len(val) == traj_len
            if attr not in traj_concat:
                traj_concat[attr] = val
            else:
                traj_concat[attr] = np.concatenate((traj_concat[attr], val), axis=0)
    return traj_concat


def get_start_end_idxs(traj_len: int, window: int, stride: int = 1, keep_last: bool = True) -> List[int]:
    '''Computes list of start and end indices given the trajectory length, window, and stride
    Args:
        traj_len [int]: trajectory length
        keep_last [bool]: whether to keep the last subtrajectory if the window does not include it
    Returns:
        start_end_idxs [List[int]]: numpy array of start & end indices, shape (num slices, 2)
    '''
    num_slices = int((traj_len - window) / stride + 1)
    start_end_idxs = [[idx * stride, idx * stride + window] for idx in range(num_slices)]
    # if the end of the trajectory has not been included, add it in
    if keep_last and (traj_len - window) % stride != 0:
        start_end_idxs.append([traj_len - window, traj_len])
    return start_end_idxs
