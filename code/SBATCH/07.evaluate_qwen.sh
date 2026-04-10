#!/bin/bash
#SBATCH --job-name=qwen-eval
#SBATCH --partition=scavenger-gpu
#SBATCH --account=dctrl-as1676
#SBATCH --gres=gpu:6000_ada_generation:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/hpc/dctrl/as1676/projects/corporate-control/code/SBATCH/logs/eval.out
#SBATCH --error=/hpc/dctrl/as1676/projects/corporate-control/code/SBATCH/logs/eval.err
#SBATCH --mail-user=as1676@duke.edu
#SBATCH --mail-type=BEGIN,FAIL,END

cd ~/dctrl_as1676/projects/corporate-control
source /hpc/dctrl/as1676/miniconda3/etc/profile.d/conda.sh
conda activate qwen-ft
export HF_HOME=/hpc/dctrl/as1676/models/hf_cache

python code/07.evaluate_qwen.py
