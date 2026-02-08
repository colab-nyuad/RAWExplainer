#!/bin/bash

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
cd $code_dir

NUM_GPUS=2
GPUS="[0,1]"
port=$(python script/find_free_port.py)
echo "Using port: $port"

SCRIPT_NAME=script/train_nbfnet.py
CONFIG_0=config/new_nbfnet/feb2026-NoFinetune/fb15k237-k-25-gnn.yaml
CONFIG_1=config/new_nbfnet/feb2026-NoFinetune/fb15k237-k-25.yaml
CONFIG_2=config/new_nbfnet/feb2026-NoFinetune/fb15k237-k-50.yaml
CONFIG_3=config/new_nbfnet/feb2026-NoFinetune/fb15k237-k-75.yaml
CONFIG_4=config/new_nbfnet/feb2026-NoFinetune/fb15k237-k-100.yaml
CONFIG_5=config/new_nbfnet/feb2026-NoFinetune/fb15k237-k-300.yaml
CONFIG_6=config/new_nbfnet/feb2026-NoFinetune/fb15k237-k-500.yaml
CONFIG_7=config/new_nbfnet/feb2026-NoFinetune/fb15k237-k-1000.yaml
echo "Finetuning NBFNet from explanation at $(date) using script: $SCRIPT_NAME"

echo "Running training for 25 edges with GNN"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
       --master_port="$port" $SCRIPT_NAME \
       -c $CONFIG_0 \
       --gpus $GPUS --use_wandb yes

echo "Running training for 25 edges"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
       --master_port="$port" $SCRIPT_NAME \
       -c $CONFIG_1 \
       --gpus $GPUS --use_wandb yes

echo "Running training for 50 edges"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
       --master_port="$port" $SCRIPT_NAME \
       -c $CONFIG_2 \
       --gpus $GPUS --use_wandb yes

echo "Running training for 75 edges"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
       --master_port="$port" $SCRIPT_NAME \
       -c $CONFIG_3 \
       --gpus $GPUS --use_wandb yes

echo "Running training for 100 edges"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
       --master_port="$port" $SCRIPT_NAME \
       -c $CONFIG_4 \
       --gpus $GPUS --use_wandb yes

echo "Running training for 300 edges"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
       --master_port="$port" $SCRIPT_NAME \
       -c $CONFIG_5 \
       --gpus $GPUS --use_wandb yes

echo "Running training for 500 edges"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
       --master_port="$port" $SCRIPT_NAME \
       -c $CONFIG_6 \
       --gpus $GPUS --use_wandb yes

echo "Running training for 1000 edges"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
       --master_port="$port" $SCRIPT_NAME \
       -c $CONFIG_7 \
       --gpus $GPUS --use_wandb yes

echo "Finetuning completed at $(date)"