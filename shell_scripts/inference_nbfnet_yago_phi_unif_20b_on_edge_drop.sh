#!/bin/bash
#SBATCH --job-name=NBFNet-YAGO-20b-RobustEval-Inf-Phi
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --time=4:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/slurm_v2_%x_%j.out
#SBATCH --error=./job_outputs/slurm_v2_%x_%j.err

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/train_nbfnet.py
CONFIG_DIR=config/nbfnet
CONFIG_NAME=yago310-20b-phi-dist-inf-on-edge-drop.yaml
cd $code_dir

# Logging
echo "Running inference script: $SCRIPT_NAME"
echo "Using Python interpreter: $python_path"
echo "Code directory: $code_dir"
echo "Conda environment path: $conda_env_path"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"
echo "Checkpoint path: /scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG/experiments/NBFNet/YAGO310/2025-09-18-18-11-22/model_epoch_0.pth"

# Single GPU
# $python_path $SCRIPT_NAME -c $CONFIG_DIR/$CONFIG_NAME --gpus [0] --use_wandb no

# DDP
port=$($python_path script/find_free_port.py)
echo "Using port: $port"
$python_path -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" $SCRIPT_NAME -c $CONFIG_DIR/$CONFIG_NAME --gpus [0,1] --use_wandb no

# Logging
echo "Inference job finished at: $(date)"
