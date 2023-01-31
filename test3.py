# import pickle
# from databoost.utils.data import find_pkl, find_h5, read_h5
# from pprint import pprint
# import numpy as np
# import os


# def print_traj_shapes(traj_batch, prefix=None):
#     for attr, val in traj_batch.items():
#         if (isinstance(val, dict)):
#             if prefix is None:
#                 print_traj_shapes(val, attr)
#             else:
#                 print_traj_shapes(val, f"{prefix}/{attr}")
#             continue
#         if prefix is None:
#             print(f"{attr} [{type(val)}]: {val.shape}")
#         else:
#             print(f"{prefix}/{attr} [{type(val)}]: {val.shape}")

# root = "/data/jullian-yapeter/DataBoostBenchmark/metaworld_rl_v2_h5"

# traj = read_h5(os.path.join(root, "assembly-v2/assembly-v2_0_0_1.h5"))

# print_traj_shapes(traj)
# print(f"num dones: {sum(traj.dones)}")
# print(f"num success: {np.any(traj.infos['success'])}")

import os
import numpy as np
from databoost.utils.data import find_pkl, read_pkl, write_h5
from databoost.utils.general import AttrDict
from typing import Dict
from tqdm import tqdm
import _pickle as pickle


n_epochs = 2
n_tasks = 50
n_steps_per_ep = 500
n_trajs = 10

root = "/home/jullian-yapeter/data/metaworld/metaworld_rl_v5"

tasks_list = [
    'stick-pull-v2',
    'faucet-open-v2',
    'faucet-close-v2',
    'stick-push-v2',
    'basketball-v2',
    'button-press-v2',
    'door-open-v2',
    'push-v2',
    'reach-wall-v2',
    'handle-press-side-v2',
    'button-press-topdown-v2',
    'hand-insert-v2',
    'handle-pull-v2',
    'shelf-place-v2',
    'pick-out-of-hole-v2',
    'pick-place-v2',
    'sweep-into-v2',
    'push-back-v2',
    'plate-slide-side-v2',
    'sweep-v2',
    'assembly-v2',
    'button-press-wall-v2',
    'disassemble-v2',
    'peg-insert-side-v2',
    'plate-slide-back-v2',
    'door-lock-v2',
    'peg-unplug-side-v2',
    'handle-pull-side-v2',
    'door-close-v2',
    'pick-place-wall-v2',
    'coffee-pull-v2',
    'hammer-v2',
    'window-close-v2',
    'plate-slide-back-side-v2',
    'door-unlock-v2',
    'window-open-v2',
    'soccer-v2',
    'push-wall-v2',
    'drawer-open-v2',
    'reach-v2',
    'button-press-topdown-wall-v2',
    'coffee-push-v2',
    'handle-press-v2',
    'drawer-close-v2',
    'box-close-v2',
    'lever-pull-v2',
    'bin-picking-v2',
    'dial-turn-v2',
    'coffee-button-v2',
    'plate-slide-v2'
]

for ep in tqdm(range(n_epochs)):
    for task in tasks_list:
        print(f"{ep}: {task}")
        # mins = []
        # maxs = []
        for traj_id in range(n_trajs):
            prev_goal = None
            for step in range(n_steps_per_ep):
                filename = os.path.join(root, f"{ep}/{task}/{traj_id}/{task}_{ep}_{traj_id}_{step}.pkl")
                try:
                    with open(filename, "rb") as f:
                        traj = None
                        traj = pickle.load(f)
                except Exception as e:
                    print(filename, "ERROR")
                # mins.append(np.min(traj["observations"]))
                # maxs.append(np.max(traj["observations"]))
                # if (traj["step_types"] != 1) or (step == n_steps_per_ep-1): prev_goal = None
                if (prev_goal is not None) and (not np.all(traj["observations"] == prev_goal)):
                    raise ValueError
                prev_goal = traj["next_observations"]
                traj = None
        # print(f"{ep}_{task}: {np.min(mins)}, {np.max(maxs)}")