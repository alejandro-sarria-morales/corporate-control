#!/bin/bash
#SBATCH --job-name=glassdoor-label
#SBATCH --partition=scavenger-gpu
#SBATCH --account=dctrl-as1676
#SBATCH --gres=gpu:6000_ada_generation:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=/hpc/dctrl/as1676/projects/corporate-control/code/SBATCH/logs/glassdoor_label.out
#SBATCH --error=/hpc/dctrl/as1676/projects/corporate-control/code/SBATCH/logs/glassdoor_label.err
#SBATCH --mail-user=as1676@duke.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --requeue

cd ~/dctrl_as1676/projects/corporate-control
source /hpc/dctrl/as1676/miniconda3/etc/profile.d/conda.sh
conda activate qwen-ft
export HF_HOME=/hpc/dctrl/as1676/models/hf_cache

python code/09.label_glassdoor.py
