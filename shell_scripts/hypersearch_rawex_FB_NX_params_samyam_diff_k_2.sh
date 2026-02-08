#!/bin/bash
#SBATCH --job-name=RAWExplainer-FB-NX-HPO-Diff-K-TrainSamples-2
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=95:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/2025-10-04-hyp/hypersearch_%x_%j.log
#SBATCH --error=./job_outputs/2025-10-04-hyp/hypersearch_%x_%j.log

#activate any environments if required
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate Ultra

# Set CUDA memory management settings to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

code_dir=/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG
HYP_SCRIPT=script_hypersearch/hypersearch_rawexplainer_fb_nx_rw_diff_2.py
OPTIM_SCRIPT=script/global_explanation_optim_diff_k_2.py
CONFIG=config/new_explain/fb15k237-rawexplainer-hyp-samyam-nx-rw-diff-k-2.yaml
TRAIN_CONFIG=config/new_nbfnet/fb15k237-finetune-samyam-nx-diff-k-2.yaml
port=$(python script/find_free_port.py)

echo "Starting hyperparameter search at $(date)"
echo "Running 51 hyperparameter combinations (0-50)"


# for i in {0..9}
# do
#     python script_hypersearch/hypersearch_rawexplainer_fb.py -c config/new_explain/fb15k237-rawexplainer-hyp-jub1.yaml --hyper_run_id="$i"
#     python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" script/global_explanation_optim.py -c config/new_explain/fb15k237-rawexplainer-hyp-jub1.yaml --hyper_search --train_config config/new_nbfnet/fb15k237-finetune-jub1.yaml
# done

for i in {0..50}
do
    echo "=========================================="
    echo "Running hyperparameter combination $i/50 at $(date)"
    echo "=========================================="
    
    python $HYP_SCRIPT -c "$CONFIG" --hyper_run_id="$i" --gpus [0,1,2,3]
    if [ $? -ne 0 ]; then
        echo "ERROR: Hyperparameter generation failed for run $i"
        continue
    fi
    
    python -m torch.distributed.launch --nproc_per_node=4 --master_port="$port" $OPTIM_SCRIPT -c "$CONFIG" --gpus [0,1,2,3] --use_wandb yes --hyper_search --train_config "$TRAIN_CONFIG"
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for run $i"
    else
        echo "Successfully completed run $i at $(date)"
    fi
done

echo "Hyperparameter search completed at $(date)"

