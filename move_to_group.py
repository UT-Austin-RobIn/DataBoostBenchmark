from databoost.envs.metaworld.config import tasks
from databoost.utils.data import find_h5
from tqdm import tqdm
import os
import random
import shutil
from pprint import pprint


src_root = "/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/large_prior/"
dest_root = "/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/"
target_dest_root = "/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/"
total_demos = 20000
num_increments = 10
num_tasks = 50
all_tasks = list(tasks.keys())
all_paths = {"success": {}, "fail": {}}

for mode in ("success", "fail"):
    for task in all_tasks:
        all_paths[mode][task] = find_h5(os.path.join(src_root, mode, task))

num_demos_per_task = total_demos // num_increments // num_tasks
print(num_demos_per_task)

for mode in ("success", "fail"):
    for group_num in tqdm(range(1, 11)):
        end_idx = group_num * num_demos_per_task
        start_idx = end_idx - num_demos_per_task

        dest_dir = os.path.join(dest_root, mode, f"{group_num}_of_{num_increments}")
        os.makedirs(dest_dir, exist_ok=True)
        selected_paths = []
        for task in all_tasks:
            selected_paths += all_paths[mode][task][start_idx : end_idx]

        for path in tqdm(selected_paths):
            shutil.copy(path, dest_dir)

        if mode == "success":
            target_dest_dir = os.path.join(target_dest_root, f"{group_num}_of_{num_increments}")
            os.makedirs(target_dest_dir, exist_ok=True)
            target_task_paths = [path for path in selected_paths if "pick-place-wall" in path]
            for path in tqdm(target_task_paths):
                shutil.copy(path, target_dest_dir)