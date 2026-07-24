#!/bin/bash
#SBATCH --job-name=qwen35b-ml-ft
#SBATCH --partition=scavenger-gpu
#SBATCH --account=dctrl-as1676
#SBATCH --gres=gpu:6000_ada_generation:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=/hpc/dctrl/as1676/projects/corporate-control/code/SBATCH/logs/ml_finetune.out
#SBATCH --error=/hpc/dctrl/as1676/projects/corporate-control/code/SBATCH/logs/ml_finetune.err
#SBATCH --mail-user=as1676@duke.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --requeue

# scavenger-gpu is preemptible (REQUEUE/GANG); --requeue + the resumable Optuna
# sqlite study means a preempted job continues its search on restart.

# Setup
cd ~/dctrl_as1676/projects/corporate-control

source /hpc/dctrl/as1676/miniconda3/etc/profile.d/conda.sh
conda activate qwen-ft

# Keep HF cache off /hpc/home (25 GB quota is the #1 cause of job failure).
export HF_HOME=/hpc/dctrl/as1676/models/hf_cache

# Build the frozen split only if it's missing; otherwise reuse the synced artifact
# so the train/test split can't drift across library versions between runs.
if [ ! -f data/trainingfinal/labelled.csv ]; then
    python code/10.prepare_multilabel_data.py
fi

python code/11.finetune_multilabel.py
