#!/bin/bash
#SBATCH --job-name=qwen35b-ml-smoke
#SBATCH --partition=scavenger-gpu
#SBATCH --account=dctrl-as1676
#SBATCH --gres=gpu:6000_ada_generation:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/hpc/dctrl/as1676/projects/corporate-control/code/SBATCH/logs/ml_smoke.out
#SBATCH --error=/hpc/dctrl/as1676/projects/corporate-control/code/SBATCH/logs/ml_smoke.err
#SBATCH --mail-user=as1676@duke.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --requeue

# Smoke run: 1 trial x 2 folds x 1 epoch. Confirms the pipeline produces non-zero F1
# before committing GPU-days to the full search. Check the log for, in order:
#   supervised tokens in example 0: 3   <- completion-only loss took effect
#   at_cap=0                            <- nothing truncated
#   trainable params well under 233M    <- attention-only LoRA
#   raw='10' parsed=(1,0) ok=True       <- the model emits parseable digits
#   malformed near zero, fold F1 > 0
# Train loss collapsing below ~0.01 again means the answer still is not being learned:
# stop and re-check the assertions rather than scaling up.

cd ~/dctrl_as1676/projects/corporate-control

source /hpc/dctrl/as1676/miniconda3/etc/profile.d/conda.sh
conda activate qwen-ft

# Keep HF cache off /hpc/home (25 GB quota is the #1 cause of job failure).
export HF_HOME=/hpc/dctrl/as1676/models/hf_cache

if [ ! -f data/trainingfinal/labelled.csv ]; then
    python code/10.prepare_multilabel_data.py
fi

export ML_N_TRIALS=1
export ML_N_FOLDS=2
export ML_EPOCHS=1
# Its own sqlite study, so smoke trials are never counted toward or selected by the
# real search; and no final training, so the production adapter is left alone.
export ML_STUDY_TAG=smoke
export ML_SKIP_FINAL=1

python code/11.finetune_multilabel.py
