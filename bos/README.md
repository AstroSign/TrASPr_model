# Bayesian Optimization for Splicing (BOS)
Official implemention of BOS. 

## 1. Getting Started

### 1.1 Weights and Biases (wandb) tracking
This repo is set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site

### 1.2 Cloning the Repo (Git Lfs)
This repository uses git lfs to store larger data files and model checkpoints. Git lfs must therefore be installed before cloning the repository. 

```Bash
conda install -c conda-forge git-lfs
```

## 2. Environment setup

We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). Also, please make sure you have at least one NVIDIA GPU with Linux x86_64 Driver Version >= 410.48 (compatible with CUDA 10.0). 

#### 2.1 Create and activate a new virtual environment

```
conda create -n traspr python=3.8
conda activate traspr
```

#### 2.2 Install the packages and other requirements for TrASPr model 

(Required)

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

cd TrASPr_MODEL
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```

#### 2.3 Install additional packages for Bayesian Optimization 
(Required)

```
pip install fire 
pip install wandb 
pip install botorch 
pip install gpytorch 
pip install pytorch-lightning
```


## 3. How to Run BOS to find sequences that optimize PSI 

Below, we provide an detials and an example command that can be used to start a BOS optimization run after the environment has been set up. 

### 3.1 Args:

To start an optimization run with BOS, run `scripts/run_bos.py` with desired command line arguments. To get a list of command line args, run the following: 

```Bash
cd scripts/

python3 run_bos.py -- --help
```

The above commands will give defaults for each arg and a description of each.

### 3.2 Example Command
Below is an example command run BOS to optimize PSI for a particular specified target tissue, with a constraint to limit the edit distance from an example starter sequence to at most 30 edits. The provided arguments are just examples, change as needed for your desired BOS run. 

```Bash
cd scripts 
python3 run_bos.py --wandb_entity $YOUR_WANDB_ENTITY --tissue $TARGET_TISSUE --minimize_psi False --edit_distance_threshold 30 --path_to_starter_sequence ../starter_sequences/dev_3.tsv --starter_sequence_id chr14_+_modulizer_nonchg_nonskip_00000100_nonchgCase_-31.65_G36M_S0fpoc --oracle_model_weights_path ../tasks/best_gtex_checkpoint --num_initialization_points 1024 --max_n_oracle_calls 100000 --bsz 10 - run - done 
```

By default, BOS will find sequences that maximize PSI for the specified target tissue. To instead find sequences that minimize psi, use `--minimize_psi True`.


### 3.3 Specifying Target Tissue

Use the following argument to specify the specific tissue for which you'd like to optimize PSI. Here $TARGET_TISSUE is the string id for your desired target tissue. 

```Bash
--tissue $TARGET_TISSUE
```

This codebase supports the following values for $TARGET_TISSUE: 

lung, heart, brain, spleen, cells_EBV_transformed_lymphocytes, K562_WT, HepG2_WT


## 4. BOS Outputs 

In addition to logging optimization results with weights and biases, BOS will also save a csv file with all sequences and corresponding PSI values found during the course of optimization. These files are saved locally in bos/save_opt_data in a subdirectory named with the unique weights and biases identifier string assigned to the run. This way the weights and baises log for the run can be easily matched up with the local csv file of all data collected. 
