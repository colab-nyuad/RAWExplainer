#!/bin/bash
#SBATCH --job-name=RAWExplainer-NELL-USE-PROP-NO
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --time=09:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/%x_%j.log
#SBATCH --error=./job_outputs/%x_%j.log


code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/global_explanation_optim.py
CONFIG_DIR=config/new_explain
CONFIG_NAME=nell995-rawexplainer-useprop-no.yaml
cd $code_dir

# Logging
echo "Running script: $SCRIPT_NAME"
echo "Using Python interpreter: $python_path"
echo "Code directory: $code_dir"
echo "Conda environment path: $conda_env_path"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# Single GPU
# $python_path $SCRIPT_NAME -c $CONFIG_DIR/$CONFIG_NAME --use_wandb yes

# Distributed
# For use_pyg_propagation = no, nbfnet/models_expl.py will comment out "assert use_pyg_propagation" (line 93)
$python_path -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 $SCRIPT_NAME -c $CONFIG_DIR/$CONFIG_NAME --use_wandb yes

# Logging
echo "Job finished at: $(date)"
