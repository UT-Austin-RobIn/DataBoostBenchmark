import databoost
from torch.utils.data import DataLoader
import numpy as np
import os

from databoost.base import DataBoostBenchmarkBase

def get_task_name_from_path(path):
    splits = path.split('/')
    task_name = splits[-1].split('_')[0]
    if 'v2' in task_name:
        task_name = task_name.replace('-v2', '')
    return task_name

def create_idx_to_path_map(dataset, fname):
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import NDArrayDecoder
    pipelines={
        'observations': [NDArrayDecoder()],
        'actions': [NDArrayDecoder()],
    }

    loader = Loader(
                fname=fname, 
                batch_size=1,
                num_workers=4,
                order=OrderOption.SEQUENTIAL,
                pipelines=pipelines,
                # indices=np.random.permutation(1347335)[:500],
            )
    
    from tqdm import tqdm
    action2idx = {}
    counter = 0
    for data in tqdm(loader):
        action = data[1][0]
        state = data[0][0]

        key = action.tobytes() + state.tobytes()
        assert key not in action2idx, "Duplicate action"
        action2idx[key] = counter
        counter += 1
    
    mapping = [-1]*len(loader)
    for i in range(len(dataset)):
        data = dataset[i]
        action = data[1]
        state = data[0]

        key = action.tobytes() + state.tobytes()
        mapping[action2idx[key]] = i
    
    print(mapping)
    # np.save("idx2path_map.npy", np.array(mapping))

def create_ordered_data_path_file(dataset, traj_index):
    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     assert data[2] == traj_index[i]
    
    # a = []
    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     path = dataset.paths[data[2]]
    #     if get_task_name_from_path(path) == 'pick-place':
    #         a.append(1)
    #     else:
    #         a.append(0)
    # a = np.array(a, dtype=np.bool)
    # print(a.shape, np.sum(a))
    # np.save('pick_place_mask.npy', a)
    # exit(0)

    with open('baseline_no-pnp-wall_horizon50_stride50.pkl', 'wb') as f:
        import pickle
        idx_to_path = {}
        for i in range(len(dataset.slices)):
            path_id, _, _ = dataset.slices[i]
            path = dataset.paths[path_id]
            # assert traj_index[i] == path_id # should be false when chunking trajectories
            idx_to_path[traj_index[i]] = path
        pickle.dump(idx_to_path, f)

def chunk_traj_index(traj_index, horizon):
    traj_id = 0
    i = 0
    while i < len(traj_index):
        val = traj_index[i]
        if len(traj_index) < i + horizon:
            traj_index[i:] = traj_id
            traj_id += 1
            i = len(traj_index)
            break
        elif traj_index[i+horizon-1] == val:
            traj_index[i: i+horizon] = traj_id
            traj_id += 1
            i += horizon
        elif traj_index[i+horizon-1] != val:
            j = i+1
            while val == traj_index[j]:
                j += 1
            traj_index[i:j] = traj_id
            traj_id += 1
            i = j
        else:
            raise NotImplementedError
    return traj_index

def write_to_ffcv(dataset, traj_index, name):
    print(np.array(traj_index).shape)
    print(np.unique(traj_index))
    print(len(np.unique(traj_index)))
    print(traj_index)

    filename = f'/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/{name}.beton'
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    np.save(f"/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/{name}_metadata.npy", np.array(traj_index))

    from ffcv.writer import DatasetWriter
    from ffcv.fields import NDArrayField, FloatField

    print(len(dataset))
    writer = DatasetWriter(filename, {
        'observations': NDArrayField(shape=(obs_dim,), dtype=np.dtype('float32')),
        'actions': NDArrayField(shape=(act_dim,), dtype=np.dtype('float32')),
    }, num_workers=1)

    writer.from_indexed_dataset(dataset)

def load_ffcv(fname):
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import NDArrayDecoder
    from ffcv.transforms import ToTensor
    pipelines={
        'observations': [NDArrayDecoder(), ToTensor()],
        'actions': [NDArrayDecoder(), ToTensor()],
    }

    loader = Loader(
                fname=fname, 
                batch_size=1,
                num_workers=4,
                order=OrderOption.QUASI_RANDOM,
                pipelines=pipelines,
                # indices=np.random.permutation(1347335)[:500],
            )
    print(len(loader))
    for data in loader:
        print(data[0].shape, data[1].shape)


if __name__ == "__main__":
    goal_conditioned=True
    mask_goals=True
    seq_len = 50
    stride = 50
    
    obs_dim = (39*(2 if goal_conditioned else 1))*seq_len
    act_dim = 4*seq_len

    # initialize environment
    benchmark = databoost.get_benchmark(benchmark_name="metaworld", mask_goal_pos=mask_goals)
    env = benchmark.get_env("pick-place-wall")
    
    cond = None 
    if mask_goals and goal_conditioned:
        cond = 'last'
    elif mask_goals and (not goal_conditioned):
        cond = 'none'
    elif (not mask_goals) and (not goal_conditioned):
        cond = 'mw'

    # for val_size in [5, 10, 20]:
    #     dataset = env._get_dataset(goal_condition=goal_conditioned, seq_len=seq_len, stride=stride, \
    #                             extra_path=[f'/home/shivin/Desktop/datamodels/data/metaworld/seed_vars/seed_{val_size}/pick-place-wall'])
    #     traj_index = np.array(dataset.task_indices)
    #     write_to_ffcv(dataset, traj_index, name=f'with_val/cond-{cond}_no-pnp-wall_traj_val-{val_size}/cond-{cond}_all_traj_val-{val_size}')
    # traj_index = chunk_traj_index(traj_index, 10)
    
    dataset = env._get_dataset(goal_condition=goal_conditioned, seq_len=seq_len, stride=stride, pad_short_traj=True)
    traj_index = np.array(dataset.task_indices)
    # traj_index = chunk_traj_index(traj_index, 10)

    # create_idx_to_path_map(dataset, fname='/home/shivin/Downloads/metaworld_no_pick_place_wall_training/metaworld_train_wo_pick-place-wall.beton')
    # create_ordered_data_path_file(dataset, traj_index)
    
    # write_to_ffcv(dataset, traj_index, name=f'cond-last_all_traj/cond-last_all_traj')
    write_to_ffcv(dataset, traj_index, name=f'baselines/baseline_cond-{cond}_all_horizon{seq_len}_stride{stride}/baseline_cond-{cond}_all_horizon{seq_len}_stride{stride}')
    # load_ffcv('/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/cond-mw_all_traj/cond-mw_all_traj.beton')

    