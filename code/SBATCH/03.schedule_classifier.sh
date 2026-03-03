#!/bin/bash
#SBATCH --job-name=roberta-train
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/roberta_%j.out
#SBATCH --error=logs/roberta_%j.err
#SBATCH --mail-type=END,FAIL

source ~/.bashrc
conda activate roberta_env

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
nvidia-smi
echo "Python: $(which python)"
echo "---"

python 03.scheduler_classifier.py

echo "Done: $(date)"