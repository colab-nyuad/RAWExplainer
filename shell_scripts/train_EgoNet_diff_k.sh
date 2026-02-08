#!/bin/bash

# Variables
code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
SCRIPT_NAME=script/instance_explanation_optim.py
MAIN_CONFIG=config/new_explain/fb15k237-egonet-benchmark-random-hops-k.yaml
FINETUNE_CONFIG=config/new_nbfnet/feb2026-NoFinetune/fb/fb15k237-egonet-fixed-k.yaml
cd $code_dir

# Loop through hops
HOPS=(1 2 3 4 5 6)
for hop in "${HOPS[@]}"
do
    echo ""
    echo "=========================================="
    echo "Starting training with hop: $hop"
    echo "=========================================="
    echo ""
    
    # Logging
    echo "Running script: $SCRIPT_NAME"
    echo "Job started at: $(date)"
    echo "Current working directory: $(pwd)"
    echo "Hop: $hop"
    
    # DDP
    port=$(python script/find_free_port.py)
    echo "Using port: $port"
    python -m torch.distributed.launch --nproc_per_node=2 \
            --master_port="$port" $SCRIPT_NAME \
            -c $MAIN_CONFIG \
            --k_hops $hop \
            --use_wandb yes --gpus [0,1] \
            --train_config $FINETUNE_CONFIG
done

echo ">>>>>> All hops (${HOPS[@]}) completed at: $(date) <<<<<<"