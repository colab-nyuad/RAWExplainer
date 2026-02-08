#!/bin/bash
#SBATCH --job-name=Job_14
#SBATCH -q bosco
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --time=95:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slamichhane@nyu.edu
#SBATCH --output=./job_outputs/2026-01-31/%x_%j.log
#SBATCH --error=./job_outputs/2026-01-31/%x_%j.log

# conda
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate Ultra

# jobs.sh
SCRIPT_A=shell_scripts/job_14a.sh
SCRIPT_B=shell_scripts/job_14b.sh
chmod +x $SCRIPT_A
chmod +x $SCRIPT_B

# Infinite loop to execute jobs.sh
while true; do
    $SCRIPT_A
    $SCRIPT_B
done