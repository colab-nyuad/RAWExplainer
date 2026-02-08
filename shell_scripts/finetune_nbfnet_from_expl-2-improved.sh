#!/bin/bash

set -euo pipefail

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
cd $code_dir

NUM_GPUS=2
GPUS="[0,1]"
port=$(python script/find_free_port.py)
echo "Using port: $port"

SCRIPT_NAME=script/train_nbfnet.py
CONFIGS=(
    # config/new_nbfnet/feb2026-NoFinetune/yago-re/yago310-gnnex-k-base.yaml
    # config/new_nbfnet/feb2026-NoFinetune/yago-re/yago310-pgex-k-base.yaml
    # config/new_nbfnet/feb2026-NoFinetune/yago-re/yago310-powerlink-k-base.yaml
    # config/new_nbfnet/feb2026-NoFinetune/yago-re/yago310-rawex-k-base.yaml

    config/new_nbfnet/feb2026-NoFinetune/yago-re/yago310-rawex-k-base.yaml
)

echo "Finetuning NBFNet from explanation at $(date) using script: $SCRIPT_NAME"

# Array of k values to iterate over
K_VALUES=(25 50 75 100 300 500 1000)
# K_VALUES=(1000)

for CONFIG_BASE in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG_BASE" .yaml)
    echo "========================================"
    echo "Using config: $CONFIG_NAME"
    echo "========================================"

    for k in "${K_VALUES[@]}"; do
        echo "Running training for $k edges"
        python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
            --master_port="$port" $SCRIPT_NAME \
            -c $CONFIG_BASE \
            --gpus $GPUS --k_edges $k --use_wandb yes
    done
done

echo "Finetuning completed at $(date)"
