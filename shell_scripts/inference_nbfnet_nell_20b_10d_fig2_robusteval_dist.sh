#!/bin/bash
#SBATCH --job-name=NBFNet-NELL-Inference-RobustEval-Dist
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=25GB
#SBATCH --time=4:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/v2_inference_%x_for_12069122.out
#SBATCH --error=./job_outputs/v2_inference_%x_for_12069122.err

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/train_nbfnet.py
CONFIG_DIR=config/nbfnet
CONFIG_NAME=nell995-20b-10d-robust-eval-dist-inference.yaml
cd $code_dir

# Logging
echo "Running inference script: $SCRIPT_NAME"
echo "Using Python interpreter: $python_path"
echo "Code directory: $code_dir"
echo "Conda environment path: $conda_env_path"
echo "Inference config: $CONFIG_DIR/$CONFIG_NAME"
echo "Model checkpoint: /scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG/experiments/NBFNet/NELL995/2025-09-19-11-26-51-phi-eval-dist/model_epoch_15.pth"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# Single GPU inference (no need for DDP during inference)
$python_path $SCRIPT_NAME -c $CONFIG_DIR/$CONFIG_NAME --gpus [0] --use_wandb no

# Logging
echo "Inference job finished at: $(date)"
