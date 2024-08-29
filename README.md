# DataBoostBenchmark
Benchmark for leveraging large unstructured datasets to augment task-specific seed datasets.

## Setup Instructions

### Setup Conda Env
```
conda create -n name_of_env python=3.9
conda activate name_of_env
```
Note: I have only tested it with python3.9 but it may work with other versions

### Setup Metaworld simulator
We use [metaworld](https://github.com/Farama-Foundation/Metaworld) simulator for the multi-task setting in this benchmark.

Setup Repository
```
git clone https://github.com/Farama-Foundation/Metaworld.git
cd Metaworld
pip install -e .
```

(IMPORTATN) Several changes have been made to the metaworld repo, so we need to use an old commit,
```
git checkout a98086a
```

#### Mujoco-py requirement
The older version of metaworld needs some care with mujoco-py.

1. Download MuJoCo Version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)
2. Extract the downloaded ```mujoco210``` directory into ```~/.mujoco/mujoco210```.
3. Run ```pip install mujoco-py``` 

### Install DataBoostBenchmark
```
git clone git@github.com:jullian-yapeter/DataBoostBenchmark.git
cd DataBoostBenchmark
git checkout datamodels

# Install correct versions of the packages to work with the old version
pip install -r requirements 
pip install -e .
```

### Download Data
The data for metaworld can be downloaded as follows,
```
gdown --no-cookies "1cWJOcRF2ixDD6h40XtmVSZB_Yf1ue3cu&confirm=t"
```
If this fails for some reason, gdown would show which link this can manually be downloaded from.

Change the env_root [here](https://github.com/jullian-yapeter/DataBoostBenchmark/blob/cd20e0aa2e85e7c6870fcc88ee529df4bd0e3ec2/databoost/envs/metaworld/config.py#L11) to point to the dataset path that was downloaded above. 
