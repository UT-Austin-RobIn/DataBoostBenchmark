from typing import Dict
from tqdm import tqdm

import os
import numpy as np

from databoost.utils.data import read_pkl, write_h5
from databoost.utils.general import AttrDict


n_epochs = 51
n_tasks = 50
n_steps_per_ep = 500
n_trajs = 10

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

traj_keys = [
    "observations",
    "actions",
    "rewards",
    "dones",
    "infos",
    "imgs"
]

root = "/data/metaworld_rl_v3"
dest_root = "/data/metaworld_rl_v3_h5"


def init_traj():
    '''Initialize an empty trajectory, preparing for data collection

    Returns:
        traj [Dict]: dict of empty attributes
    '''
    traj = AttrDict()
    for attr in traj_keys:
        traj[attr] = [] if attr not in ("info", "infos") else {}
    return traj


def add_to_traj(
        traj: AttrDict,
        ob: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        info: Dict,
        im: np.ndarray = None):
    '''helper function to append a step's results to a trajectory dictionary

    Args:
        traj [AttrDict]: dictionary with keys {
            observations, actions, rewards, dones, infos, imgs}
        ob [np.ndarray]: env-specific observation
        act [np.ndarray]: env-specific action
        rew [float]: reward; float
        done [bool]: done flag
        info [Dict]: task-specific info
        im [np.ndarray]: rendered image after the step
    '''
    traj.observations.append(ob)
    traj.actions.append(act)
    traj.rewards.append(rew)
    traj.dones.append(done)
    for attr in info:
        if attr not in traj.infos:
            traj.infos[attr] = []
        traj.infos[attr].append(info[attr])
    if im is not None:
        traj.imgs.append(im)


def traj_to_numpy(traj: AttrDict) -> AttrDict:
    '''convert trajectories attributes into numpy arrays

    Args:
        traj [AttrDict]: dictionary with keys {obs, acts, rews, dones, infos, ims}
    Returns:
        traj_numpy [AttrDict]: trajectory dict with attributes as numpy arrays
    '''
    traj_numpy = init_traj()
    for attr in traj:
        if attr not in ("info", "infos"):
            traj_numpy[attr] = np.array(traj[attr]).squeeze()
        else:
            for info_attr in traj.infos:
                traj_numpy.infos[info_attr] = np.array(
                    traj.infos[info_attr]).squeeze()
    return traj_numpy


made_tasks = set()
for ep in tqdm(range(n_epochs)):
    for task in tqdm(tasks_list):
        for traj_id in range(n_trajs):
            traj_h5 = init_traj()
            path_num = 0
            for step in tqdm(range(n_steps_per_ep)):
                filename = os.path.join(
                    root, f"{ep}/{task}/{traj_id}/{task}_{ep}_{traj_id}_{step}.pkl")
                traj = read_pkl(filename)
                # import pdb; pdb.set_trace()
                traj["env_infos"].pop("task_name")
                done = (traj["step_types"] in (2, 3)) or (
                    step == n_steps_per_ep-1)
                add_to_traj(
                    traj_h5,
                    traj["observations"][0][:39],
                    traj["actions"][0],
                    0,
                    done,
                    traj["env_infos"]
                )
                if done:
                    traj_h5 = traj_to_numpy(traj_h5)
                    dest_dir = os.path.join(dest_root, task)
                    if task not in made_tasks:
                        os.makedirs(dest_dir, exist_ok=True)
                        made_tasks.add(task)
                    dest_path = os.path.join(
                        dest_dir, f"{task}_{ep}_{traj_id}_{path_num}.h5")
                    write_h5(traj_h5, dest_path)
                    traj_h5 = init_traj()
                    path_num += 1
