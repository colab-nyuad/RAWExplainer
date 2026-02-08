#!/bin/bash
#SBATCH --job-name=NBFNet-NELL-0b-Test
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --time=2:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/slurm_%x_%j_test.out
#SBATCH --error=./job_outputs/slurm_%x_%j_test.err

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/train_nbfnet.py
CONFIG_DIR=config/nbfnet
CONFIG_NAME=nell995-0b-test.yaml
cd $code_dir

# Logging
echo "Running script: $SCRIPT_NAME"
echo "Using Python interpreter: $python_path"
echo "Code directory: $code_dir"
echo "Conda environment path: $conda_env_path"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# Single GPU
# $python_path $SCRIPT_NAME -c $CONFIG_DIR/$CONFIG_NAME --gpus [0] --use_wandb no

# Logging
echo "Job finished at: $(date)"
