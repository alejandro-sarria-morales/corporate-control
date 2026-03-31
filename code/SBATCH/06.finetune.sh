#!/bin/bash
#SBATCH --job-name=qwen-ft-test
#SBATCH --partition=gpu-common
#SBATCH --account=dctrl-as1676
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/finetune.out
#SBATCH --error=logs/finetune.err
#SBATCH --mail-user=as1676@duke.edu
#SBATCH --mail-type=FAIL,END


# Setup
cd ~/dctrl_as1676/projects/corporate-control

source ~/.bashrc
conda activate qwen-ft
export HF_HOME=/hpc/dctrl/as1676/models/hf_cache

python code/06.finetune.py
