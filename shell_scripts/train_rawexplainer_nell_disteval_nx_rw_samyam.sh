#!/bin/bash
#SBATCH --job-name=RAWE-NELL-NX-RW-Diff
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --time=19:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/2025-09-30/slurm_explainer_%x_%j_re_with_phi_dist.log
#SBATCH --error=./job_outputs/2025-09-30/slurm_explainer_%x_%j_re_with_phi_dist.log

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/global_explanation.py
MAIN_CONFIG=config/new_explain/nell995-rawexplainer-nx-rw.yaml
cd $code_dir

# Logging
echo "Running script: $SCRIPT_NAME"
echo "Using Python interpreter: $python_path"
echo "Code directory: $code_dir"
echo "Conda environment path: $conda_env_path"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# ONE GPU
# $python_path $SCRIPT_NAME -c $CONFIG_DIR/$CONFIG_NAME --gpus [0] --use_wandb yes

# DDP
port=$($python_path script/find_free_port.py)
$python_path -m torch.distributed.launch --nproc_per_node=3 --master_port="$port" $SCRIPT_NAME -c $MAIN_CONFIG --use_wandb yes --gpus [0,1,2]

# Logging
echo "Job finished at: $(date)"