#!/bin/bash
echo ""
echo "========== (Script A) =========="
echo ""

# Variables
code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
# SCRIPT_NAME=script/global_explanation_optim.py
SCRIPT_NAME=script/instance_explanation_optim.py
# SCRIPT_NAME=script/train_nbfnet.py
MAIN_CONFIG=config/new_explain/feb2026-noFinetune/yago310-bfsnode-benchmark-noEvalTrain-headsonly.yaml
FINETUNE_CONFIG=config/new_nbfnet/feb2026-NoFinetune/yago-re/yago310-randomwalk-fixed-k.yaml
cd $code_dir

# Logging
echo "Running script: $SCRIPT_NAME"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

#$ DDP
port=$(python script/find_free_port.py)
echo "Using port: $port"
python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port="$port" $SCRIPT_NAME \
        -c $MAIN_CONFIG \
        --use_wandb yes --gpus [0,1] \
        --train_config $FINETUNE_CONFIG

#$ Training EgoNet with different hops
# bash shell_scripts/train_EgoNet_diff_k.sh

# Logging
echo "Job finished at: $(date)"

# # load
echo "Running load.py"
python misc/load.py

# --- Stop Logic ---
REFERENCE_FILE="shell_scripts/stop4.txt"
echo "Checking contents of $REFERENCE_FILE"
while true
do
    # Check for 'exit' in any line
    if grep -Fxq "exit" "$REFERENCE_FILE"; then
        echo "'exit' found in $REFERENCE_FILE. Exiting loop."
        exit 0
    fi

    sleep 1
done