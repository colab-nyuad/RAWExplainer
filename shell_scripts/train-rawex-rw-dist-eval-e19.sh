#!/bin/bash
echo ""
echo "========== (Script A) =========="
echo ""

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate Ultra

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
SCRIPT_NAME=script/global_explanation_optim.py
MAIN_CONFIG=config/new_explain/nell995-rawexplainer-rw-dist-eval-e19.yaml
FINETUNE_CONFIG=config/new_nbfnet/nell995-finetune-samyam.yaml
cd $code_dir

# Logging
echo "Running script: $SCRIPT_NAME"
echo "Code directory: $code_dir"
echo "Job started at: $(date)"
echo "Current working directory: $(pwd)"

# DDP
port=$(python script/find_free_port.py)
python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port="$port" $SCRIPT_NAME -c $MAIN_CONFIG \
        --use_wandb yes --gpus [0,1] \
        --train_config $FINETUNE_CONFIG

# # load
echo "Running load.py"
python misc/load.py