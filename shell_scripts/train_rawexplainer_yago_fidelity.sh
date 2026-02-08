#!/bin/bash

# Variables
code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
SCRIPT_NAME=script/global_explanation_optim.py
MAIN_CONFIG=config/new_explain/feb2026-noFinetune/yago310-rawex-k25-1k-expl-gen-fidelity.yaml
FINETUNE_CONFIG=config/new_nbfnet/yago310-finetune-samyam-re-4percent-re-kall-fidelity.yaml
cd $code_dir

# Logging
echo "========================================"
echo "RAWExplainer on NELL995 with Fidelity+ Metric"
echo "========================================"
echo "Running script: $SCRIPT_NAME"
echo "Config: $MAIN_CONFIG"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"
echo "========================================"

#$ DDP
port=$(python script/find_free_port.py)
echo "Using port: $port"
python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port="$port" $SCRIPT_NAME \
        -c $MAIN_CONFIG \
        --use_wandb yes --gpus [0,1] \
        --train_config $FINETUNE_CONFIG

