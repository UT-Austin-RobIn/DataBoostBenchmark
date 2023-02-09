# CURATE: A Benchmark for Data Curation in Policy Learning
#### [[Project Website]](https://sites.google.com/view/curate-benchmark/)


## Installation Instructions

1. Install Conda
https://docs.conda.io/en/latest/miniconda.html)

2. Clone CURATE repository and set up CurateEnv (Python 3.7) environment.
```
cd DataBoostBenchmark
conda env create -f curate_env.yml
pip install -e DataBoostBenchmark
```

3. Install dependencies
```
# install MuJoCo
Please follow the instructions here...
https://github.com/openai/mujoco-py#install-mujoco

# install Meta-World environment
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld

# install Language Table environment
git clone git@github.com:google-research/language-table.git
pip install -e language-table

# install garage
pip install garage

# install R3M
git clone git@github.com:facebookresearch/r3m.git
pip install -e r3m

# install CLIP
pip install git+https://github.com/openai/CLIP.git
```

4. Download the CURATE dataset
```
mkdir CurateDataset
cd CurateDataset

# download Meta-World dataset
gdown 1cWJOcRF2ixDD6h40XtmVSZB_Yf1ue3cu

# download Robot Kitchen dataset
gdown ....

# download Language Table dataset
gdown 1cWJOcRF2ixDD6h40XtmVSZB_Yf1ue3cu

unzip *
```

## Demo Script

This provides an introduction to the CURATE framework:
```
python3 demo.py
```
It illustrates CURATES dataloaders for both the small task-specific and large unstructured datasets, as well as how to evaluate policies on our benchmark.

## Baseline training script
```
python3 databoost/scripts/train_policy.py
```

## Baseline evaluation script
```
python3 databoost/scripts/evaluate_policy.py
```

## Files and Directories
* `demo.py`: An introduction to our benchmark framework
* `base.py`: Implements base functionality of our benchmark and environments
* `envs/`: configures our environments [`Meta-World`](https://github.com/Farama-Foundation/Metaworld), [`Language Table`](https://github.com/google-research/language-table) and `Robot Kitchen`
* `models/`: Baseline BC and IQL models
* `scripts/`: Includes `train_policy.py` and `evaluate_policy.py`, main training and testing scripts for our baseline
* `utils/`: Geneal utility function


## Datasets
Install gdown
```
pip install gdown
```
* `Metaworld`: contains seed dataset of `pick-place-wall` demos (`seed/`) and unstructured multi-task dataset (`prior/`)
```
gdown 1cWJOcRF2ixDD6h40XtmVSZB_Yf1ue3cu
```
* `Language Table`: contains seed dataset of `separate` demos (`seed/`) and unstructured multi-task dataset (`prior/`)
'''
gdown 
'''
* `Robot Kitchen`: contains play dataset of robot 
```
gdown 1bIxib5smogg_eK_6OC8AKH5cRhdDQruX
```
