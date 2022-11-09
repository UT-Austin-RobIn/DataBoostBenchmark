import json
import os
from typing import Any, List, Dict

import h5py

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
            data[key] = F[key][()]
    return data


def write_h5(data: Dict, dest_path: str):
    '''Copy selected trajectory of a dataset to the destination directory as an h5 file.
    Args:
        data [AttrDict]: dictionary of data to be written to an h5 file
        dest_path [str]: path of destination file
    '''
    with h5py.File(dest_path, "w") as F:
        for attr in data:
            F[attr] = data[attr]


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
