#!/bin/bash
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate Ultra

# Variables
code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
SCRIPT_NAME=script/train_nbfnet.py
CONFIG_NAME=config/nbfnet/fb15k237-04b-010d-4max-dist-lambda-E0-inf-E9.yaml
cd $code_dir

# Logging
echo "Running script: $SCRIPT_NAME"
echo "Code directory: $code_dir"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# DDP
port=$(python script/find_free_port.py)
echo "Using port: $port"
python -m torch.distributed.launch --nproc_per_node=2 \
            --master_port="$port" \
            $SCRIPT_NAME -c $CONFIG_NAME \
            --gpus [0,1] \
            --use_wandb yes \
            --d 0.20

# Logging
echo "Job finished at: $(date)"
