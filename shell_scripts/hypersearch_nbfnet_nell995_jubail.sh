#!/bin/bash
#SBATCH --job-name=NBFNet-NELL-HyperSearch-Re
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=95:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/slurm_%x_%j_Sept16_2nd.out
#SBATCH --error=./job_outputs/slurm_%x_%j_Sept16_2nd.log

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate Ultra

# Set CUDA memory management settings to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
SCRIPT_NAME=script_hypersearch/hypersearch_nbfnet.py
NBFNET_SCRIPT=script/train_nbfnet.py
CONFIG="config/nbfnet/nell995-hypersearch.yaml"
port=$(python script/find_free_port.py)

for i in {0..30}
do
    python $SCRIPT_NAME -c "$CONFIG" --hyper_run_id="$i"
    python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" $NBFNET_SCRIPT -c "$CONFIG" --hyper_search
done

