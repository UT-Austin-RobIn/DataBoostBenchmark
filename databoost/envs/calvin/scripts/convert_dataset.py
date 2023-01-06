import os
import numpy as np
from typing import Dict

from tqdm import tqdm

from databoost.utils.general import AttrDict
from databoost.utils.data import write_h5
from databoost.envs.calvin.utils import render
from databoost.envs.calvin.custom import CalvinEnv


def npz_files(dir_path):
    paths = (os.path.join(dir_path, filename) for filename in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, filename)) and
             filename.endswith(".npz"))
    return sorted(paths)


def get_scene_id(scene_intervals, ep_id):
    for scene_id, scene_interval in scene_intervals.items():
        start, end = scene_interval
        if ep_id >= start and ep_id <= end:
            return scene_id
    return None


def get_traj_id(ep_start_end_ids, ep_id):
    for traj_id, traj_interval in enumerate(ep_start_end_ids):
        start, end = traj_interval
        if ep_id >= start and ep_id <= end:
            return traj_id
    return None


def init_traj() -> AttrDict:
    traj = {}
    for attr in [
            "observations",
            "actions",
            "rewards",
            "dones",
            "infos",
            "imgs"
        ]:
        traj[attr] = [] if attr != "infos" else {}
    return AttrDict(traj)


def add_to_traj(traj: AttrDict,
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
            traj_numpy[attr] = np.array(traj[attr])
        else:
            for info_attr in traj.infos:
                traj_numpy.infos[info_attr] = np.array(traj.infos[info_attr])
    return traj_numpy


def get_skill(skill_data, ep_id):
    def get_skill_recur(skill_data, low, high, ep_id): 
        # Check base case
        if high >= low:
            mid = (high + low) // 2
            # If element is present at the middle itself
            start, end, skill = skill_data[mid]
            if ep_id >= start and ep_id <= end:
                return skill
            # If element is smaller than mid, then it can only
            # be present in left subarray
            elif ep_id < start:
                return get_skill_recur(skill_data, low, mid - 1, ep_id)
            # Else the element can only be present in right subarray
            else:
                return get_skill_recur(skill_data, mid + 1, high, ep_id)
        else:
            # Element is not present in the array
            return "None"
    return get_skill_recur(skill_data, 0, len(skill_data)-1, ep_id)


if __name__=="__main__":
    DEST_DIR = "/data/jullian-yapeter/DataBoostBenchmark/calvin/data/prior"

    # load in episode interval and skill info
    data_root = "/data/lucys/skill/calvin/dataset/task_ABCD_A"
    phase = "training"
    data_dir = os.path.join(data_root, phase)
    scene_intervals = np.load(os.path.join(data_root,
                                           "training/scene_info.npy"),
                              allow_pickle=True).item()
    # scene_intervals is a dict of {A: [0, 611098], B: [611099, 1210008], ... }
    scene_intervals = {scene.split("_")[-1]: interval
                       for scene, interval in scene_intervals.items()}
    ep_start_end_ids = np.load(os.path.join(data_dir, "ep_start_end_ids.npy"))
    print(ep_start_end_ids)
    skill_data = np.load(os.path.join(data_dir,
                                      "lang_annotations/auto_lang_ann.npy"),
                         allow_pickle=True).item()
    skill_strings = skill_data["language"]["task"]
    skill_intervals = skill_data["info"]["indx"]
    skill_data = [(*interval, skill) for interval, skill in zip(skill_intervals, skill_strings)]
    # skill_data is a sorted list of tuples (start_idx, end_idx, skill_string)
    skill_data = sorted(skill_data)

    # iterate through episode step files and build the episode trajectory
    traj_data = init_traj()
    data_files = npz_files(data_dir)
    curr_ep_id = int(os.path.splitext(os.path.split(data_files[0])[-1].split("_")[-1])[0])
    curr_traj_id = get_traj_id(ep_start_end_ids, curr_ep_id)
    curr_scene_id = get_scene_id(scene_intervals, curr_ep_id)
    # env = CalvinEnv("", scene=curr_scene_id)  # set to empty task, just for rendering purposes
    for ep_step_path in tqdm(data_files):
        # convert "path/to/dir/episode_0001234.npz" to 1234
        ep_id = int(os.path.splitext(os.path.split(ep_step_path)[-1].split("_")[-1])[0])
        # identify the traj id, scene, skill
        traj_id = get_traj_id(ep_start_end_ids, ep_id)
        scene_id = get_scene_id(scene_intervals, ep_id)
        skill = get_skill(skill_data, ep_id)
        # write current trajectory when transition to next trajectory detected
        if traj_id != curr_traj_id:
            traj_numpy = traj_to_numpy(traj_data)
            write_h5(traj_numpy, os.path.join(DEST_DIR, f"traj_{phase}_{curr_traj_id}.h5"))
            curr_traj_id = traj_id
            traj_data = init_traj()
            # env = CalvinEnv("", scene=scene_id)
        # get step data
        ep_step_data = np.load(ep_step_path)
        # env.reset(robot_obs=ep_step_data["robot_obs"],
        #           scene_obs=ep_step_data["scene_obs"])
        ob = np.concatenate([ep_step_data["robot_obs"],
                             ep_step_data["scene_obs"]])
        # im = render(env)
        act = ep_step_data["rel_actions"]
        rew = 0
        done = 0
        # _, _, _, info = env.step(act)
        info = {"scene": scene_id.encode("utf-8"), "skill": skill.encode("utf-8")}
        add_to_traj(traj_data, ob, act, rew, done, info, None)
    # write the last trajectory
    traj_numpy = traj_to_numpy(traj_data)
    write_h5(traj_numpy, os.path.join(DEST_DIR, f"traj_{curr_traj_id}.h5"))
