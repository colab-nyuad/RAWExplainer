#!/bin/bash
#SBATCH --job-name=NBFNet-NELL-0.1FR-Phi-0b-Train
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --time=03:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/2025-09-24/slurm_%x_%j_train.log
#SBATCH --error=./job_outputs/2025-09-24/slurm_%x_%j_train.log

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/train_nbfnet.py
CONFIG_NAME=config/nbfnet/nell995-0b-fr010.yaml
cd $code_dir

# Logging
echo "Running script: $SCRIPT_NAME"
echo "Using Python interpreter: $python_path"
echo "Code directory: $code_dir"
echo "Conda environment path: $conda_env_path"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# Single GPU
# $python_path $SCRIPT_NAME -c $CONFIG_DIR/$CONFIG_NAME --gpus [0] --use_wandb yes

# DDP
port=$($python_path script/find_free_port.py)
echo "Using port: $port"
$python_path -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" $SCRIPT_NAME -c $CONFIG_NAME --gpus [0,1] --use_wandb yes

# Logging
echo "Job finished at: $(date)"
