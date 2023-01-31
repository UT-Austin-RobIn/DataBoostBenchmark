import pickle
from databoost.utils.data import find_pkl, find_h5, read_h5
from pprint import pprint
import numpy as np
from tqdm import tqdm


total_length = 0
mins = []
maxs = []
filenames = []
for e in range(4, 20, 2):
    filenames += find_h5(f"/data/jullian-yapeter/DataBoostBenchmark/metaworld_rl_v3_h5/{e}")
for filename in tqdm(filenames):
    traj = read_h5(filename)
    total_length += len(traj["observations"])
    # mins.append(np.min(traj["observations"]))
    # maxs.append(np.max(traj["observations"]))
print(f"{total_length}")
# count = {}
# for i in range(3):
#     count[i] = {}
#     pkl_files = find_pkl(f"/data/jullian-yapeter/DataBoostBenchmark/metaworld_rl/{i}")
#     for pkl_file in pkl_files:
#         with open(pkl_file, "rb") as f:
#             path = pickle.load(f)
#         task_name = path["env_infos"]["task_name"][0]
#         if task_name not in count[i]: count[i][task_name] = 0
#         count[i][task_name] += 1
# print(pprint(count))

# filename = find_h5("/data/karl/data/table_sim/prior_data")[0]
# traj = read_h5(filename)
# print(traj)

# prev_goal = np.random.rand(1, 89)
# pkl_files = find_pkl()
# print(len(pkl_files))
# exit(0)
# with open("/data/jullian-yapeter/DataBoostBenchmark/metaworld_rl/v2/0/assembly-v2/assembly-v2_0_0.pkl", "rb") as f:
#     path = pickle.load(f)
#     import pdb; pdb.set_trace()
#     target_goal = path["next_observations"]
# for idx in range(401, 25201, 400):
# for idx in range(200):
# for filename in pkl_files:
# for filename in find_pkl("/data/jullian-yapeter/DataBoostBenchmark/metaworld_rl/v2"):
    # with open(f"/data/jullian-yapeter/DataBoostBenchmark/metaworld_rl/0/assembly-v2/assembly-v2_0_{idx}.pkl", "rb") as f:
    # filename = f"/data/jullian-yapeter/DataBoostBenchmark/metaworld_rl/0/assembly-v2/assembly-v2_0_{idx}.pkl"
    # filename = f"/data/jullian-yapeter/DataBoostBenchmark/metaworld_rl_v2/0/assembly-v2/1/assembly-v2_0_1_{idx}.pkl"
    # with open(filename, "rb") as f:
    #     path = pickle.load(f)
    #     if np.all(path["observations"] == prev_goal) and idx != 0:
            # print(path["observations"][0, -53:-50])
        #     print("YES")
        # prev_goal = path["next_observations"]
        # if (path["step_types"] != 1):
        #     print(filename, path["step_types"])
    # if filename != "/data/jullian-yapeter/DataBoostBenchmark/metaworld_rl/0/assembly-v2/assembly-v2_0_400.pkl":
    # if idx != 401:
    # if np.all(path["observations"] == target_goal):
    #     print("YES")
    # else:
    #     print("NO")
    #     print(path["step_types"])
    #     import pdb; pdb.set_trace()
    # target_goal = path["next_observations"]
    # prev_goal = path["next_observations"]
    # if "pick-place-wall-v2" in pkl_file:
    # with open(pkl_file, "rb") as f:
    #     path = pickle.load(f)
    # if not (np.all(path["observations"][0, -3:] == prev_goal)):
    #     print(path["observations"][0, -3:], prev_goal)
    # prev_goal = path["next_observations"][0, -3:]
    # if(==2)):
    # if(path["env_infos"]["success"]==1 and path["step_types"] != 1):
    #     print(path["step_types"])
    # if(path["step_types"] in [2]):
    #     print(pkl_file)