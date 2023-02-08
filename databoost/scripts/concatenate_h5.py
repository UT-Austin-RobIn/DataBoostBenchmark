

import os
from databoost.utils.data import find_h5, read_h5, write_h5
import numpy as np
from tqdm import tqdm


seed_filenames = find_h5("/home/jullian-yapeter/data/DataBoostBenchmark/metaworld/dataset/seed")
prior_filenames = find_h5("/home/jullian-yapeter/data/DataBoostBenchmark/metaworld/dataset/autonomous")
prior_filenames += find_h5("/home/jullian-yapeter/data/DataBoostBenchmark/metaworld/dataset/demonstration")
prior_filenames = [pf for pf in prior_filenames if "pick-place-wall" not in pf]

dest_filename = "/home/jullian-yapeter/data/boosted_data/metaworld/pick-place-wall/NoTarget-Curated/nt_curated_NoRL_metaworld.h5"

#### Hand curation
accept_list = ("push-back", "pick-place", "hand-insert", "push")
print(f"Curating Dataset from original {len(prior_filenames)}")
print(f"Using acceptance list: {accept_list}")
def hand_curate(fp):
    if "seed" in fp:
        return True
    for task in accept_list:
        if task in fp:
            return True
    return False
prior_filenames = [fp for fp in prior_filenames if hand_curate(fp)]
print(f"Curated to {len(prior_filenames)}")
####

unprocessed = set()

def print_traj_shapes(traj_batch, prefix=None):
    for attr, val in traj_batch.items():
        if (isinstance(val, dict)):
            if prefix is None:
                print_traj_shapes(val, attr)
            else:
                print_traj_shapes(val, f"{prefix}/{attr}")
            continue
        if prefix is None:
            print(f"{attr} [{type(val)}]: {val.shape}")
        else:
            print(f"{prefix}/{attr} [{type(val)}]: {val.shape}")

def concatenate_traj(dataset, traj):
    for attr, val in traj.items():
        if isinstance(val, np.ndarray):
            if "goal_" in attr:
                orig_attr = attr.replace("goal_", "")
                assert len(val.shape) + 1 == len(traj[orig_attr].shape)
                val = np.repeat(val[None], len(traj[orig_attr]), axis=0)
            if len(val) == len(traj["observations"]):
                dataset[attr] = np.concatenate((dataset[attr], val), axis=0) \
                    if dataset.get(attr) is not None else val
                continue
        unprocessed.add(attr)
    # return dataset

dataset = {}

for filename in tqdm(seed_filenames):
    traj = read_h5(filename)
    # Seed preprocessing
    traj["terminals"] = np.zeros_like(traj["dones"])
    traj["rewards"] = np.zeros_like(traj["dones"])
    traj["seed"] = np.ones_like(traj["dones"])
    traj["terminals"][-1] = 1.
    traj["rewards"][-1] = 1.
    # attrs = list(traj.keys())
    # for attr in attrs:
    #     if "goal_" not in attr and isinstance(traj[attr], np.ndarray) and len(traj[attr]) > 0:
    #         traj[f"goal_{attr}"] = traj[attr][-1]
    concatenate_traj(dataset, traj)

for filename in tqdm(prior_filenames):
    traj = read_h5(filename)
    # Prior preprocessing
    traj["terminals"] = np.zeros_like(traj["dones"])
    traj["rewards"] = np.zeros_like(traj["dones"])
    traj["seed"] = np.zeros_like(traj["dones"])
    traj["terminals"][-1] = 1.
    concatenate_traj(dataset, traj)

# add empty elem at the very end for easy slicing purposes
for attr in dataset:
    empty_elem = np.zeros_like(dataset[attr][0])[None]
    dataset[attr] = np.concatenate((dataset[attr], empty_elem), axis=0)

print(dataset.keys())
print("unprocessed: ", unprocessed)

print_traj_shapes(dataset)
write_h5(dataset, dest_filename)