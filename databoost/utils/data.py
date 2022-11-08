from typing import Dict

import h5py


def write_h5(data: Dict, dest_path: str):
    '''Copy selected trajectory of a dataset to the destination directory as an h5 file.
    Args:
        data [AttrDict]: dictionary of data to be written to an h5 file
        dest_path [str]: path of destination file
    '''
    with h5py.File(dest_path, "w") as F:
        for attr in data:
            F[attr] = data[attr]
