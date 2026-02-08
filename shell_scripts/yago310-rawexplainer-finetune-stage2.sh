#!/bin/bash

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate Ultra

# Set CUDA memory management settings to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
port=$(python script/find_free_port.py)
cd $code_dir

echo "Starting training at $(date)"

echo "Running training for 10 edges"
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port="$port" script/train_nbfnet.py \
       -c config/new_nbfnet/yago310-finetune-samyam-stage2-10.yaml \
       --gpus [0,1] --use_wandb yes

echo "Running training for 25 edges"
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port="$port" script/train_nbfnet.py \
       -c config/new_nbfnet/yago310-finetune-samyam-stage2-25.yaml \
       --gpus [0,1] --use_wandb yes

echo "Running training for 100 edges"
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port="$port" script/train_nbfnet.py \
       -c config/new_nbfnet/yago310-finetune-samyam-stage2-100.yaml \
       --gpus [0,1] --use_wandb yes

echo "Running training for 300 edges"
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port="$port" script/train_nbfnet.py \
       -c config/new_nbfnet/yago310-finetune-samyam-stage2-300.yaml \
       --gpus [0,1] --use_wandb yes

echo "Running training for 500 edges"
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port="$port" script/train_nbfnet.py \
       -c config/new_nbfnet/yago310-finetune-samyam-stage2-500.yaml \
       --gpus [0,1] --use_wandb yes

# load
echo "Running load.py"
python misc/load.py

echo "Training completed at $(date)"