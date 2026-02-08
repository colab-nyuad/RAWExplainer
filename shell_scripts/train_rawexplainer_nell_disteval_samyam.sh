#!/bin/bash
#SBATCH --job-name=RAWE-NELL-Phi-Eval-Dist
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --time=25:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/slurm_v2_explainer_%x_%j.out
#SBATCH --error=./job_outputs/slurm_v2_explainer_%x_%j.log

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
conda_env_path=/scratch/sl9030/conda-envs/Ultra
python_path=$conda_env_path/bin/python
SCRIPT_NAME=script/global_explanation.py
MAIN_CONFIG=config/new_explain/nell995-rawexplainer-jubail-samyam.yaml
FINETUNE_CONFIG=config/nbfnet/nell995-rawexplainer-finetune20-jubail-samyam.yaml
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
$python_path -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" $SCRIPT_NAME -c $MAIN_CONFIG --use_wandb yes --gpus [0,1] --train_config $FINETUNE_CONFIG

# Logging
echo "Job finished at: $(date)"