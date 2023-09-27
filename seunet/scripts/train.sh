#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=64:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=sparseunet
#SBATCH --output=./outputs/train/job_%j.out
#SBATCH --mail-user=benjamin.eckhardt@ut.ee

eval "$(micromamba shell hook --shell $(basename "$SHELL"))"
micromamba activate yaroslav

nvidia-smi

python main.py --job_id $SLURM_JOB_ID