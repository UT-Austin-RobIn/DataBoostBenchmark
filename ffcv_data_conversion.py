import databoost
from torch.utils.data import DataLoader
import numpy as np

from databoost.base import DataBoostBenchmarkBase

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
    np.save("idx2path_map.npy", np.array(mapping))

def create_ordered_data_path_txt_file(dataset):
    with open('/home/shivin/Desktop/datamodels/data/metaworld/ordered_data_paths.txt', 'w') as f:
        for i in range(len(dataset)):
            data = dataset[i]
            path = dataset.paths[data[2]]
            f.write(path + '\n')

    # data_paths = []
    # with open('/home/shivin/Desktop/datamodels/data/metaworld/ordered_data_paths.txt', 'r') as f:
    #     for line in f:
    #         path = line.strip()
    #         data_paths.append(path)

    # for i, path in enumerate(dataset.paths):
    #     assert path == data_paths[i], f"{path} != {data_paths[i]}"
    #     print(path)

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
    np.save(f"/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/{name}_metadata.npy", np.array(traj_index))

    from ffcv.writer import DatasetWriter
    from ffcv.fields import NDArrayField, FloatField

    print(len(dataset))
    writer = DatasetWriter(f'/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/{name}.beton', {
        'observations': NDArrayField(shape=(78,), dtype=np.dtype('float32')),
        'actions': NDArrayField(shape=(4,), dtype=np.dtype('float32')),
    }, num_workers=8)

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
    # initialize environment
    benchmark = databoost.get_benchmark(benchmark_name="metaworld")
    env = benchmark.get_env("pick-place-wall")

    dataset = env._get_dataset()

    traj_index = []
    for data in dataset:
        traj_index.append(data[2])
    traj_index = np.array(traj_index)
    # traj_index = chunk_traj_index(traj_index, 10)
    
    # create_idx_to_path_map(dataset, fname='/home/shivin/Downloads/metaworld_no_pick_place_wall_training/metaworld_train_wo_pick-place-wall.beton')
    # create_ordered_data_path_txt_file(dataset)

    write_to_ffcv(dataset, traj_index, name='hand-curated_wo_pick-place-wall')
    # load_ffcv('/home/shivin/Desktop/datamodels/data/metaworld/ffcv_training_data/metaworld_no_pick_place_wall_training/metaworld_train_wo_pick-place-wall.beton')

    