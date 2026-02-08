#!/bin/bash

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
cd $code_dir

NUM_GPUS=2
GPUS="[0,1]"
port=$(python script/find_free_port.py)
echo "Using port: $port"

SCRIPT_NAME=script/train_nbfnet.py
CONFIG_1=config/new_nbfnet/feb2026-NoFinetune/wn/wn18rr-pg-k-1000.yaml
echo "Finetuning NBFNet from explanation at $(date) using script: $SCRIPT_NAME"

echo "Running training for 1000 edges"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
       --master_port="$port" $SCRIPT_NAME \
       -c $CONFIG_1 \
       --gpus $GPUS --use_wandb no

echo "Finetuning completed at $(date)"