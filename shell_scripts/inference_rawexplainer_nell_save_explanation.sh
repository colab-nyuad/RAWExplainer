#!/bin/bash
#SBATCH --job-name=RAWE-NELL-Inf-Save-Explanation
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --time=15:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/2025-10-01/inference_explainer_%x_%j.log
#SBATCH --error=./job_outputs/2025-10-01/inference_explainer_%x_%j.log

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/global_explanation_optim.py
INFERENCE_CONFIG=config/new_explain/nell995-rawexplainer-samyam-save-explanation.yaml
cd $code_dir

# Logging
echo "Running inference script: $SCRIPT_NAME"
echo "Using Python interpreter: $python_path"
echo "Code directory: $code_dir"
echo "Conda environment path: $conda_env_path"
echo "Inference config: $INFERENCE_CONFIG"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# Single GPU inference (no need for DDP during inference)
$python_path $SCRIPT_NAME -c $INFERENCE_CONFIG --use_wandb yes --gpus [0]

# Logging
echo "Inference job finished at: $(date)"
