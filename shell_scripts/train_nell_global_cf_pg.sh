#!/bin/bash

# Variables
code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
SCRIPT_NAME=script/global_explanation_optim.py
FINETUNE_CONFIG=config/new_nbfnet/cf/nell_cf.yaml

# List of main configs to loop through
list_configs=(
    "config/new_explain/cf/nell_pgex.yaml"
)

cd $code_dir

# Logging
echo "========================================"
echo "PGExplainer CF on NELL995"
echo "========================================"
echo "Running script: $SCRIPT_NAME"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"
echo "========================================"

# Loop through each config
for MAIN_CONFIG in "${list_configs[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing config: $MAIN_CONFIG"
    echo "========================================"
    
    #$ DDP
    port=$(python script/find_free_port.py)
    echo "Using port: $port"
    python -m torch.distributed.launch --nproc_per_node=2 \
            --master_port="$port" $SCRIPT_NAME \
            -c $MAIN_CONFIG \
            --use_wandb yes --gpus [0,1] \
            --train_config $FINETUNE_CONFIG
    
    echo "Completed config: $MAIN_CONFIG"
    echo ""
    echo ""
done

echo ""
echo "========================================"
echo "All configs completed at: $(date)"
echo "========================================"

