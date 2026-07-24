"""
Fine-tune Qwen3.5-35B-A3B (LoRA, Unsloth) for the two-dimension classifier:
each review gets schedule_related and job_control_related in {0,1}, emitted as two
digits "<schedule><jobctrl>".

Adapted from 06.finetune.py (binary schedule classifier). Differences:
  - Target is two digits instead of one; loss covers both.
  - Cross-validation is stratified on the 4-way state key (2*schedule + job_control),
    so plain StratifiedKFold stays multi-label-aware without a new dependency.
  - Optuna objective is the *schedule* F1 (the primary/headline dimension), with mean
    macro-F1 kept as the tie-breaker (schedule saturates, so the tie-break is what
    actually selects a model that is also good at the rare job_control dimension).
  - Optional oversampling of the rare states (job-control / both / none) in the TRAIN
    split only, to give the minority signal more weight.
  - Resumable Optuna study (sqlite) so a preempted scavenger-gpu job continues its
    search instead of restarting from scratch.

Run from the repo root: `python code/11.finetune_multilabel.py`
"""

import os
import sys
import json
import gc
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import optuna
from optuna.trial import TrialState
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for metrics_multilabel
from metrics_multilabel import parse_two_digit

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

# ============================================================
# Configuration
# ============================================================
MODEL_NAME  = "Qwen/Qwen3.5-35B-A3B"
MAX_SEQ_LEN = 512
DATA_CSV    = "data/trainingfinal/labelled.csv"
PROMPT_FILE = "code/prompts/multilabel_system_prompt.txt"
MODEL_SLUG  = MODEL_NAME.split("/")[-1]
RESULTS_DIR = f"models/cv_results_multilabel/{MODEL_SLUG}"
FINAL_DIR   = f"models/finetuned/{MODEL_SLUG}_multilabel"

N_FOLDS  = 5
N_TRIALS = 5
PATIENCE = 3

# Oversampling of rare states in the TRAIN split only (never val/test).
# Keys are the 4-way state code (2*schedule + job_control): 0=none,1=jobctrl,2=sched,3=both.
OVERSAMPLE = True
OVERSAMPLE_FACTORS = {0: 3, 1: 3, 3: 3}  # sched-only (2) left at 1x

with open(PROMPT_FILE, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()


def format_example(doc, sched, ctrl):
    """Chat-template example with the two-digit answer supervised."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{doc}<|im_end|>\n"
        f"<|im_start|>assistant\n{int(sched)}{int(ctrl)}<|im_end|>"
    )


def format_prompt(doc):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{doc}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def maybe_oversample(df):
    """Replicate rare-state rows in a TRAIN dataframe. No-op on val/test."""
    if not OVERSAMPLE:
        return df
    parts = [df]
    for state, factor in OVERSAMPLE_FACTORS.items():
        if factor > 1:
            sub = df[df["strat"] == state]
            parts.extend([sub] * (factor - 1))
    return pd.concat(parts, ignore_index=True)


def make_dataset(df):
    return Dataset.from_dict({
        "text": [
            format_example(r["review_text"], r["schedule_related"], r["job_control_related"])
            for _, r in df.iterrows()
        ]
    })


def compute_fold_metrics(model, tokenizer, val_df):
    """Greedy two-digit inference on val_df; return per-dimension + macro F1."""
    model.eval()
    ps, pc, malformed = [], [], 0
    for _, row in val_df.iterrows():
        prompt = format_prompt(row["review_text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=4, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        s, c, ok = parse_two_digit(text)
        malformed += (not ok)
        ps.append(s)
        pc.append(c)
        del inputs, out
    ts = val_df["schedule_related"].tolist()
    tc = val_df["job_control_related"].tolist()
    sched_f1 = f1_score(ts, ps, pos_label=1, zero_division=0)
    ctrl_f1 = f1_score(tc, pc, pos_label=1, zero_division=0)
    return {
        "schedule_f1": float(sched_f1),
        "jobcontrol_f1": float(ctrl_f1),
        "macro_f1": float(0.5 * (sched_f1 + ctrl_f1)),
        "malformed": int(malformed),
    }


def _load_lora(config):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    cache_dir = os.path.join(os.getcwd(), "unsloth_compiled_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        load_in_16bit=False,
        full_finetuning=False,
        device_map="auto",
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    model = FastModel.get_peft_model(
        model,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.0,  # MoE ParamWrapper (expert weights) require dropout=0
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=123,
        max_seq_length=MAX_SEQ_LEN,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    return model, tokenizer


def _teardown(model, tokenizer, trainer):
    trainer.model = None
    trainer.optimizer = None
    trainer.lr_scheduler = None
    for cb in trainer.callback_handler.callbacks:
        if hasattr(cb, "model"):
            cb.model = None
    del model, tokenizer, trainer
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); gc.collect()


def train_fold(train_df, val_df, config, fold_idx, output_dir):
    """Train one CV fold; return the metrics dict on the best-loss checkpoint."""
    print(f"\n{'='*60}")
    print(f"  Fold={fold_idx+1}/{N_FOLDS}  |  rank={config['lora_rank']}, "
          f"alpha={config['lora_alpha']}, lr={config['learning_rate']:.2e}")
    print(f"  Train: {len(train_df)} (oversampled), Val: {len(val_df)}")
    print(f"{'='*60}\n")

    model, tokenizer = _load_lora(config)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=make_dataset(train_df),
        eval_dataset=make_dataset(val_df),
        args=SFTConfig(
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            num_train_epochs=5,
            learning_rate=config["learning_rate"],
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=50,
            eval_accumulation_steps=2,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            output_dir=output_dir,
            seed=3407,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            prediction_loss_only=True,
        ),
        dataset_text_field="text",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )
    trainer.train()

    metrics = compute_fold_metrics(model, tokenizer, val_df)
    print(f"\n  Fold metrics: schedule_f1={metrics['schedule_f1']:.4f}  "
          f"jobcontrol_f1={metrics['jobcontrol_f1']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
          f"malformed={metrics['malformed']}  |  stopped at epoch {trainer.state.epoch:.2f}")

    fold_adapter_dir = os.path.join(output_dir, "best_adapter")
    os.makedirs(fold_adapter_dir, exist_ok=True)
    model.save_pretrained(fold_adapter_dir)
    tokenizer.save_pretrained(fold_adapter_dir)

    _teardown(model, tokenizer, trainer)
    return metrics


def train_final(train_df, config):
    """Retrain on the full train split with the best config; save to FINAL_DIR."""
    train_df = maybe_oversample(train_df)
    print(f"\n{'='*60}")
    print(f"  FINAL TRAINING on full train split ({len(train_df)} rows, oversampled)")
    print(f"  r={config['lora_rank']}, alpha={config['lora_alpha']}, lr={config['learning_rate']:.2e}")
    print(f"{'='*60}\n")

    model, tokenizer = _load_lora(config)
    os.makedirs(FINAL_DIR, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=make_dataset(train_df),
        args=SFTConfig(
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            num_train_epochs=5,
            learning_rate=config["learning_rate"],
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="no",
            save_strategy="no",
            output_dir=FINAL_DIR,
            seed=3407,
            dataloader_num_workers=0,
            dataset_num_proc=1,
        ),
        dataset_text_field="text",
    )
    trainer.train()
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"\n  Final model saved to {FINAL_DIR}")
    _teardown(model, tokenizer, trainer)


# ============================================================
# Main: Optuna hyperparameter search with stratified K-fold CV
# ============================================================
print("Loading data...")
df = pd.read_csv(DATA_CSV)
train_pool = df[df["set"] == 1].reset_index(drop=True)  # 75% train; set==0 is the held-out test
print(f"Total: {len(df)}  |  train pool: {len(train_pool)}  |  held-out test: {int((df['set']==0).sum())}")
print("Train-pool state counts:", train_pool["strat"].value_counts().sort_index().to_dict())

os.makedirs(RESULTS_DIR, exist_ok=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=123)


def objective(trial):
    rank = trial.suggest_categorical("r", [4, 8, 16, 32])
    lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
    alpha_mode = trial.suggest_categorical("alpha_mode", ["r", "2r"])
    alpha = rank if alpha_mode == "r" else 2 * rank
    config = {"lora_rank": rank, "lora_alpha": alpha, "learning_rate": lr}

    print(f"\n{'*'*60}\n  Trial {trial.number}  |  r={rank}, alpha={alpha}, lr={lr:.2e}\n{'*'*60}")

    sched_f1s, macro_f1s, ctrl_f1s = [], [], []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_pool, train_pool["strat"])):
        tr_df = maybe_oversample(train_pool.iloc[tr_idx])
        va_df = train_pool.iloc[va_idx]  # never oversampled
        fold_dir = os.path.join(RESULTS_DIR, f"trial_{trial.number}", f"fold_{fold_idx}")
        m = train_fold(tr_df, va_df, config, fold_idx, fold_dir)
        sched_f1s.append(m["schedule_f1"])
        ctrl_f1s.append(m["jobcontrol_f1"])
        macro_f1s.append(m["macro_f1"])

    # keep only the best (by schedule F1) fold's adapter
    best_fold = int(np.argmax(sched_f1s))
    for fold_idx in range(N_FOLDS):
        if fold_idx != best_fold:
            d = os.path.join(RESULTS_DIR, f"trial_{trial.number}", f"fold_{fold_idx}")
            if os.path.exists(d):
                shutil.rmtree(d)

    trial.set_user_attr("fold_schedule_f1", sched_f1s)
    trial.set_user_attr("fold_jobcontrol_f1", ctrl_f1s)
    trial.set_user_attr("mean_macro_f1", float(np.mean(macro_f1s)))
    trial.set_user_attr("mean_jobcontrol_f1", float(np.mean(ctrl_f1s)))
    mean_sched = float(np.mean(sched_f1s))
    print(f"\n  Trial {trial.number} done  |  mean_schedule_f1={mean_sched:.4f}  "
          f"mean_macro_f1={np.mean(macro_f1s):.4f}  mean_jobctrl_f1={np.mean(ctrl_f1s):.4f}")
    return mean_sched  # primary objective: schedule F1


# Resumable study so scavenger-gpu preemption doesn't discard search progress.
storage = f"sqlite:///{os.path.join(RESULTS_DIR, 'optuna_study.db')}"
study = optuna.create_study(
    direction="maximize", study_name="multilabel_lora",
    storage=storage, load_if_exists=True,
)
n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
remaining = max(0, N_TRIALS - n_complete)
print(f"Optuna: {n_complete} complete trial(s) found, running {remaining} more.")
if remaining:
    study.optimize(
        objective, n_trials=remaining,
        catch=(torch.cuda.OutOfMemoryError, NotImplementedError, RuntimeError, ValueError),
    )

# ============================================================
# Select best config (schedule F1 primary, macro F1 tie-break) and retrain
# ============================================================
print("\n" + "=" * 60 + "\nOPTUNA SEARCH RESULTS\n" + "=" * 60)
completed = [t for t in study.trials if t.value is not None]
for t in sorted(completed, key=lambda x: x.value, reverse=True):
    print(f"  trial={t.number}  mean_schedule_f1={t.value:.4f}  "
          f"mean_macro_f1={t.user_attrs.get('mean_macro_f1', 0):.4f}  params={t.params}")

best_trial = max(completed, key=lambda t: (round(t.value, 4), t.user_attrs.get("mean_macro_f1", 0.0)))
best_rank = best_trial.params["r"]
best_config = {
    "lora_rank": best_rank,
    "lora_alpha": best_rank if best_trial.params["alpha_mode"] == "r" else 2 * best_rank,
    "learning_rate": best_trial.params["lr"],
}
print(f"\nBest: trial={best_trial.number}  schedule_f1={best_trial.value:.4f}  "
      f"macro_f1={best_trial.user_attrs.get('mean_macro_f1', 0):.4f}  config={best_config}")

if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)
train_final(train_pool, best_config)

results_log = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "n_folds": N_FOLDS,
    "n_trials": N_TRIALS,
    "oversample": OVERSAMPLE,
    "oversample_factors": OVERSAMPLE_FACTORS,
    "objective": "mean schedule F1 (macro-F1 tie-break)",
    "best_trial": best_trial.number,
    "best_params": best_trial.params,
    "best_mean_schedule_f1": best_trial.value,
    "best_mean_macro_f1": best_trial.user_attrs.get("mean_macro_f1"),
    "all_trials": [
        {
            "trial": t.number,
            "params": t.params,
            "mean_schedule_f1": t.value,
            "fold_schedule_f1": t.user_attrs.get("fold_schedule_f1", []),
            "fold_jobcontrol_f1": t.user_attrs.get("fold_jobcontrol_f1", []),
            "mean_macro_f1": t.user_attrs.get("mean_macro_f1"),
        }
        for t in study.trials
    ],
}
with open(os.path.join(RESULTS_DIR, "cv_results.json"), "w") as f:
    json.dump(results_log, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_DIR}/cv_results.json")
print(f"Final model saved to {FINAL_DIR}\nDone!")
