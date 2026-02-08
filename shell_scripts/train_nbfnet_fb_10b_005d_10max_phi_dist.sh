#!/bin/bash
#SBATCH --job-name=NBFNet-FB-005b-10d-10max-Phi-Dist
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --time=95:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/2025-10-02/slurm_%x_%j.log
#SBATCH --error=./job_outputs/2025-10-02/slurm_%x_%j.log

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/train_nbfnet.py
CONFIG_NAME=config/nbfnet/fb15k237-10b-005d-10max-phi-dist.yaml

# Logging
echo "Running script: $SCRIPT_NAME"
echo "Using Python interpreter: $python_path"
echo "Code directory: $code_dir"
echo "Conda environment path: $conda_env_path"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# Single GPU
# $python_path $SCRIPT_NAME -c $CONFIG_NAME --gpus [0] --use_wandb yes

# DDP
port=$($python_path script/find_free_port.py)
echo "Using port: $port"
$python_path -m torch.distributed.launch --nproc_per_node=3 --master_port="$port" $SCRIPT_NAME -c $CONFIG_NAME --gpus [0,1,2] --use_wandb yes

# Logging
echo "Job finished at: $(date)"
