from databoost.utils.data import find_h5, read_h5, write_h5
import numpy as np
from tqdm import tqdm


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
                # val = np.concatenate((traj[orig_attr], repeated_goal), axis=-1)
            if len(val) == len(traj["observations"]):
                dataset[attr] = np.concatenate((dataset[attr], val), axis=0) \
                    if dataset.get(attr) is not None else val
                continue
        unprocessed.add(attr)
    return dataset

dataset = {}

seed_filenames = find_h5("/home/jullian-yapeter/data/DataBoostBenchmark/metaworld/dataset/seed")
for filename in tqdm(seed_filenames):
    traj = read_h5(filename)
    # Seed preprocessing
    traj["terminals"] = np.zeros_like(traj["dones"])
    traj["rewards"] = np.zeros_like(traj["dones"])
    traj["terminals"][-1] = 1.
    traj["rewards"][-1] = 1.
    attrs = list(traj.keys())
    for attr in attrs:
        if "goal_" not in attr and isinstance(traj[attr], np.ndarray) and len(traj[attr]) > 0:
            traj[f"goal_{attr}"] = traj[attr][-1]
    dataset = concatenate_traj(dataset, traj)

prior_filenames = find_h5("/home/jullian-yapeter/data/boosted_data/metaworld/pick-place-wall/BC/data/retrieved")
for filename in tqdm(prior_filenames):
    traj = read_h5(filename)
    # Prior preprocessing
    traj["terminals"] = np.zeros_like(traj["dones"])
    traj["rewards"] = np.zeros_like(traj["dones"])
    traj["terminals"][-1] = 1.
    dataset = concatenate_traj(dataset, traj)

print(dataset.keys())
print("unprocessed: ", unprocessed)

print_traj_shapes(dataset)
write_h5(dataset, "/home/jullian-yapeter/data/bc_boosted_metaworld.h5")