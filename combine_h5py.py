import numpy as np
from databoost.utils.data import find_h5, read_h5, write_h5

def recursively_copy_h5py(d, f):
    for k in f:
        if k == 'imgs':
            continue
        if hasattr(f[k], "keys") and callable(f[k].keys):
            d[k] = recursively_copy_h5py(d[k], f[k]) if k in d else recursively_copy_h5py({}, f[k])
        else:
            d[k] = np.concatenate((d[k], f[k]), axis=0) if k in d else np.array(f[k])
    return d

def recursively_add_dummy(d):
    for k in d:
        if hasattr(d[k], "keys") and callable(d[k].keys):
            d[k] = recursively_add_dummy(d[k])
        else:
            d[k] = np.concatenate((d[k], np.zeros_like(d[k][0])[None]), axis=0)
    return d

dataset_dir = ['/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/seed/pick-place-wall']# , 
                # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/prior']

all_data = {}
for cur_dataset_dir in dataset_dir:
    for i, file in enumerate(find_h5(cur_dataset_dir)):
        f = read_h5(file)
        if i%10==0: print(i)
        all_data = recursively_copy_h5py(all_data, f)

recursively_add_dummy(all_data)
all_data['dense_rewards'] = all_data['rewards']
all_data['rewards'] = np.copy(all_data['dones'])
for k in all_data:
    if k != 'infos':
        print(k, all_data[k][-2:])
print()

print(np.where(all_data['dones']))
write_h5(all_data, '/home/sdass/boosting/data/pick_place_wall/seed/seed.h5')