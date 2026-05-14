#!/bin/bash
#SBATCH --job-name=qwen-ft-test
#SBATCH --partition=h200ea
#SBATCH --account=dctrl-as1676
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/finetune.out
#SBATCH --error=logs/finetune.err
#SBATCH --mail-user=as1676@duke.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --requeue


# Setup
cd ~/dctrl_as1676/projects/corporate-control

source /hpc/dctrl/as1676/miniconda3/etc/profile.d/conda.sh
conda activate qwen-ft

export HF_HOME=/hpc/dctrl/as1676/models/hf_cache

python code/06.finetune.py
