import cv2
import time
import databoost
from databoost.models.bc import TanhGaussianBCPolicy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from databoost.base import DataBoostBenchmarkBase

if __name__ == "__main__":
    
    benchmark_name = "metaworld"
    task_name = "pick-place-wall"
    experiment_method = "R3M"
    policy_class = TanhGaussianBCPolicy
    goal_condition = True

    # initialize environment
    benchmark = databoost.get_benchmark(benchmark_name=benchmark_name)
    env = benchmark.get_env(task_name)

    # data loader
    dataloader_configs = {
        "batch_size": 500,
        "seq_len": 1,
        "shuffle": True,
        "load_imgs": False,
        "goal_condition": goal_condition
    }
    dataset = env._get_dataset()

    # traj_index = []
    # for data in dataset:
    #     traj_index.append(data[2])

    # print(np.array(traj_index).shape)
    # print(np.unique(traj_index))
    # print(len(np.unique(traj_index)))
    # np.save("../data/metaworld/traj_index.npy", np.array(traj_index))

    from ffcv.writer import DatasetWriter
    from ffcv.fields import NDArrayField, FloatField

    writer = DatasetWriter('metaworld_val.beton', {
        'observations': NDArrayField(shape=(78,), dtype=np.dtype('float32')),
        'actions': NDArrayField(shape=(4,), dtype=np.dtype('float32')),
    }, num_workers=8)

    writer.from_indexed_dataset(dataset)

    # from ffcv.loader import Loader, OrderOption
    # from ffcv.fields.decoders import NDArrayDecoder
    # from ffcv.transforms import ToTensor
    # pipelines={
    #     'observations': [NDArrayDecoder(), ToTensor()],
    #     'actions': [NDArrayDecoder(), ToTensor()],
    # }

    # loader = Loader(
    #             fname='../data/metaworld/metaworld_train.beton', 
    #             batch_size=500,
    #             num_workers=4,
    #             order=OrderOption.QUASI_RANDOM,
    #             pipelines=pipelines,
    #             indices=np.random.permutation(1347335)[:500],
    #         )
    # print(len(loader))
    # for data in loader:
    #     print(data[0].shape, data[1].shape)