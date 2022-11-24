import json
import os
from typing import Any, List, Dict, Tuple

import h5py
import numpy as np

from databoost.utils.general import AttrDict


def find_h5(dir: str) -> List[str]:
    '''Get list of h5 file paths.

    Args:
        dir [str]: path to directory of interest
    Returns:
        file_paths [List[str]]: sorted list of h5 file paths within dir
    '''
    file_paths = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith((".h5", ".hdf5")):
                file_paths.append(os.path.join(root, file))
    file_paths.sort()
    return file_paths


def read_h5(path: str) -> Dict:
    '''Read h5 file into dictionary
    Args:
        path [str]: path to the h5 file
    Returns:
        data [Dict]: data of h5 file
    '''
    def unpack_h5_recurse(h5_data):
        data = AttrDict()
        for key in h5_data.keys():
            if hasattr(h5_data[key], "keys") and callable(h5_data[key].keys):
                data[key] = unpack_h5_recurse(h5_data[key])
            else:
                data[key] = h5_data[key][()]
        return data

    with h5py.File(path) as F:
        data = unpack_h5_recurse(F)
    return data


def write_h5(data: Dict, dest_path: str):
    '''Copy selected trajectory of a dataset to the destination directory as an h5 file.

    Args:
        data [AttrDict]: dictionary of data to be written to an h5 file
        dest_path [str]: path of destination file
    '''
    def pack_h5_recurse(h5_file, data, prefix=None):
        for attr, val in data.items():
            if isinstance(val, dict):
                pack_h5_recurse(h5_file, val,
                                prefix=f"{prefix}/{attr}" if prefix is not None \
                                else attr)
            else:
                if prefix is None:
                    h5_file[attr] = val
                else:
                    h5_file[f"{prefix}/{attr}"] = val

    with h5py.File(dest_path, "w") as F:
        pack_h5_recurse(F, data)


def read_json(path: str) -> Any:
    '''Read JSON file into dictionary.

    Args:
        path [str]: path to the JSON file
    Returns:
        data [Any]: data of JSON file
    '''
    with open(path, "r") as F:
        return json.load(F)


def write_json(obj: Any, dest_path: str):
    '''Write JSON file to destination path.

    Args:
        obj [Any]: data to be written to JSON
        dest_path [str]: destination path of the JSON file
    '''
    with open(dest_path, "w") as F:
        json.dump(obj, F)


def concatenate_traj_data(trajs: Tuple[AttrDict]) -> AttrDict:
    ''' Given a tuple of attribute dictionaries each representing a trajectory,
    concatenate all the trajectories to a single attribute dictionary (as if
    one trajectory).

    Args:
        trajs [Tuple[AttrDict]]: tuple of trajectories to be concatenated
    Returns:
        traj_concat [AttrDict]: attribute dictionary of concatenated trajectories
    '''
    def concat_traj_data_recurse(traj_concat, traj, traj_len):
        for attr, val in traj.items():
            if isinstance(val, dict):
                if attr not in traj_concat:
                    traj_concat[attr] = {}
                concat_traj_data_recurse(traj_concat[attr], val, traj_len)
                continue
            assert isinstance(val, np.ndarray), f"attribute is of type {type(val)}"
            assert len(val) == traj_len
            if attr not in traj_concat:
                traj_concat[attr] = val
            else:
                traj_concat[attr] = np.concatenate((traj_concat[attr], val), axis=0)

    traj_concat = AttrDict()
    for traj in trajs:
        traj_len = len(traj.observations)
        concat_traj_data_recurse(traj_concat, traj, traj_len)
    return traj_concat


def get_start_end_idxs(traj_len: int,
                       window: int,
                       stride: int = 1,
                       keep_last: bool = True) -> List[int]:
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
