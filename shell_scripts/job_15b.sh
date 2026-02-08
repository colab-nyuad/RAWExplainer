#!/bin/bash
echo ""
echo "========== (Script B) =========="
echo ""

# Variables
code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
SCRIPT_NAME=script/global_explanation_optim.py
# SCRIPT_NAME=script/instance_explanation_optim.py
# SCRIPT_NAME=script/train_nbfnet.py
MAIN_CONFIG=config/new_nbfnet/feb2026-NoFinetune/nell/nell995-pgex-k1000-expl-gen.yaml
FINETUNE_CONFIG=config/new_nbfnet/nell995-finetune-jub4-samyam.yaml
cd $code_dir

# Logging
echo "Running script: $SCRIPT_NAME"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

#$ DDP
# port=$(python script/find_free_port.py)
# echo "Using port: $port"
# python -m torch.distributed.launch --nproc_per_node=2 \
#         --master_port="$port" $SCRIPT_NAME \
#         -c $MAIN_CONFIG \
#         --use_wandb yes --gpus [0,1] \
#         --train_config $FINETUNE_CONFIG

#$ Separate training script
bash shell_scripts/train_yago_instance_cf_gnn.sh

# Logging
echo "Job finished at: $(date)"

# # load
echo "Running load.py"
python misc/load.py

# --- Stop Logic ---
REFERENCE_FILE="shell_scripts/stop15.txt"
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