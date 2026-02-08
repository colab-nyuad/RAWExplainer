#!/bin/bash
#SBATCH --job-name=NBFNet-YAGO-10max-0.05d-Phi-Eval-Dist
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB
#SBATCH --time=95:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/slurm_v2_%x_%j_RobustEval_Dist.out
#SBATCH --error=./job_outputs/slurm_v2_%x_%j_RobustEval_Dist.err

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/train_nbfnet.py
CONFIG_DIR=config/nbfnet
CONFIG_NAME=yago310-20b-10d-robust-eval-dist.yaml
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
port=$(python script/find_free_port.py)
echo "Using port: $port"
$python_path -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" $SCRIPT_NAME -c $CONFIG_DIR/$CONFIG_NAME --gpus [0,1] --use_wandb yes

# Logging
echo "Job finished at: $(date)"
