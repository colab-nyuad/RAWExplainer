#!/bin/bash
#SBATCH --job-name=NBFNet-NELL-Finetune-HPO
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=95:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/2025-09-24/hypersearch_%x_%j.log
#SBATCH --error=./job_outputs/2025-09-24/hypersearch_%x_%j.log

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate Ultra

# Set CUDA memory management settings to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
SCRIPT_NAME=script_hypersearch/hypersearch_nbfnet_nell_finetune_on_trained.py
NBFNET_SCRIPT=script/train_nbfnet.py
CONFIG="config/nbfnet/nell995-hypersearch-nbfnet-finetune-on-trained.yaml"
port=$(python script/find_free_port.py)

echo "Starting hyperparameter search at $(date)"
echo "Running 31 hyperparameter combinations (0-30)"

for i in {0..30}
do
    echo "=========================================="
    echo "Running hyperparameter combination $i/30 at $(date)"
    echo "=========================================="
    
    python $SCRIPT_NAME -c "$CONFIG" --hyper_run_id="$i"
    if [ $? -ne 0 ]; then
        echo "ERROR: Hyperparameter generation failed for run $i"
        continue
    fi
    
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" $NBFNET_SCRIPT -c "$CONFIG" --hyper_search
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for run $i"
    else
        echo "Successfully completed run $i at $(date)"
    fi
done

echo "Hyperparameter search completed at $(date)"

