"""
Minimal HPC smoke test for 06.finetune.py.

Runs 1 Optuna trial × 2 folds × 1 epoch on 40 training examples.
Takes ~5-10 minutes on the cluster. Use this to verify that:
  - val_f1 is printed after each fold
  - cv_results.json contains fold_f1s and best_mean_val_f1
  - Optuna study direction is maximize

Submit as a short interactive or batch job before launching the full run.
"""

import os
import json
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from datetime import datetime
import shutil
import optuna
import torch

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

# ============================================================
# Configuration — same as 06.finetune.py
# ============================================================
MODEL_NAME  = "Qwen/Qwen3.5-9B"
MAX_SEQ_LEN = 512
DATA_CSV    = "data/training_set.csv"
MODEL_SLUG  = MODEL_NAME.split("/")[-1]
RESULTS_DIR = f"models/cv_results/{MODEL_SLUG}_smoke"
FINAL_DIR   = f"models/finetuned/{MODEL_SLUG}_smoke"

# ── Smoke overrides ──────────────────────────────────────────
N_FOLDS   = 2
N_TRIALS  = 1
PATIENCE  = 1
N_SMOKE_EXAMPLES = 40
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a research assistant classifying job reviews.\n"
    "Determine whether the review contains any mention of work schedule.\n\n"
    "Mentions of schedule include:\n"
    "  - Working hours (long hours, short hours, specific shifts)\n"
    "  - Overtime (mandatory or voluntary)\n"
    "  - Flexibility or rigidity of hours\n"
    "  - Availability requirements (on-call, weekends, holidays)\n"
    "  - Stability or predictability of working hours\n\n"
    "Here are some examples of a review mentioning schedule:\n"
    "  1. 'long hours and a lot of work'\n"
    "  2. 'they offer flexible hours and the staff is very nice.'\n"
    "  3. 'not enough hours a week regardless if you are willing to work more.'\n\n"
    "Here are some examples of a review not mentioning schedule:\n"
    "  1. 'i have been working at alta resources for sunrun part-time for more than a year "
    "the brea office is a tight family, everyone became friends and free lunch on thursdays.'\n"
    "  2. 'people are boring and the workplace conversations are limited. some people are not "
    "very good at their jobs and they\\'re allowed to stay way too long.'\n"
    "  3. 'career movement, leadership not diversed enough'\n\n"
    "Reply with exactly one character: 1 if the review mentions schedule, 0 if not.\n"
    "Do not add any explanation or punctuation."
)


def format_example(doc, label):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{doc}<|im_end|>\n"
        f"<|im_start|>assistant\n{int(label)}<|im_end|>"
    )


def format_prompt(doc):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{doc}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def compute_fold_f1(model, tokenizer, val_df):
    model.eval()
    preds = []
    for _, row in val_df.iterrows():
        prompt = format_prompt(row["doc"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=3, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        preds.append(1 if text.startswith("1") else 0)
    return f1_score(val_df["label"].tolist(), preds, zero_division=0)


def train_fold(train_dataset, val_dataset, config, fold_idx, output_dir, val_df):
    print(f"\n{'='*60}")
    print(f"  Fold={fold_idx+1}/{N_FOLDS}  |  rank={config['lora_rank']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']}, lr={config['learning_rate']:.2e}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"{'='*60}\n")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

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
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=123,
        max_seq_length=MAX_SEQ_LEN,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            num_train_epochs=1,          # smoke: 1 epoch only
            learning_rate=config["learning_rate"],
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=10,               # smoke: eval frequently
            eval_accumulation_steps=2,
            save_strategy="steps",
            save_steps=10,               # smoke: save frequently
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

    val_f1 = compute_fold_f1(model, tokenizer, val_df)
    print(f"\n  Val F1: {val_f1:.4f}  |  stopped at epoch: {trainer.state.epoch:.2f}")

    fold_adapter_dir = os.path.join(output_dir, "best_adapter")
    os.makedirs(fold_adapter_dir, exist_ok=True)
    model.save_pretrained(fold_adapter_dir)
    tokenizer.save_pretrained(fold_adapter_dir)

    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return val_f1


# ============================================================
# Main
# ============================================================
print("Loading data...")
df = pd.read_csv(DATA_CSV)
train_df_full = (
    df[df["set"] == 1]
    .reset_index(drop=True)
    .head(N_SMOKE_EXAMPLES)   # smoke: tiny subset
)
print(f"Smoke subset: {len(train_df_full)} examples  (label distribution: {train_df_full['label'].value_counts().to_dict()})")

os.makedirs(RESULTS_DIR, exist_ok=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=123)


def objective(trial):
    rank    = trial.suggest_categorical("r", [4, 8, 16])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.1])
    lr      = trial.suggest_float("lr", 1e-4, 3e-4, log=True)
    alpha   = rank

    config = {
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "learning_rate": lr,
    }

    print(f"\n{'*'*60}")
    print(f"  Trial {trial.number}  |  r={rank}, alpha={alpha}, dropout={dropout}, lr={lr:.2e}")
    print(f"{'*'*60}")

    fold_f1s = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df_full, train_df_full["label"])):
        train_df = train_df_full.iloc[train_idx]
        val_df   = train_df_full.iloc[val_idx]

        train_dataset = Dataset.from_dict({
            "text": [format_example(row["doc"], row["label"]) for _, row in train_df.iterrows()]
        })
        val_dataset = Dataset.from_dict({
            "text": [format_example(row["doc"], row["label"]) for _, row in val_df.iterrows()]
        })

        fold_output_dir = os.path.join(RESULTS_DIR, f"trial_{trial.number}", f"fold_{fold_idx}")

        f1 = train_fold(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            fold_idx=fold_idx,
            output_dir=fold_output_dir,
            val_df=val_df,
        )
        fold_f1s.append(f1)

    best_fold_idx = int(np.argmax(fold_f1s))
    for fold_idx in range(N_FOLDS):
        if fold_idx != best_fold_idx:
            fold_dir = os.path.join(RESULTS_DIR, f"trial_{trial.number}", f"fold_{fold_idx}")
            if os.path.exists(fold_dir):
                shutil.rmtree(fold_dir)

    trial.set_user_attr("fold_f1s", fold_f1s)
    trial.set_user_attr("best_fold_idx", best_fold_idx)
    mean_f1 = float(np.mean(fold_f1s))
    print(f"\n  Trial {trial.number} done  |  mean_val_f1={mean_f1:.4f}  |  folds: {[f'{s:.4f}' for s in fold_f1s]}")
    return mean_f1


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, catch=(torch.cuda.OutOfMemoryError, NotImplementedError, RuntimeError))

print("\n" + "=" * 60)
print("SMOKE TEST RESULTS")
print("=" * 60)

for t in sorted([t for t in study.trials if t.value is not None], key=lambda x: x.value, reverse=True):
    print(f"  trial={t.number}  |  mean_val_f1={t.value:.4f}  |  params={t.params}")

best_trial    = study.best_trial
best_fold_f1s = best_trial.user_attrs["fold_f1s"]
best_fold_idx = int(np.argmax(best_fold_f1s))
best_adapter_path = os.path.join(RESULTS_DIR, f"trial_{best_trial.number}", f"fold_{best_fold_idx}", "best_adapter")

print(f"\nBest config: trial={best_trial.number}, params={best_trial.params} (mean_val_f1={best_trial.value:.4f})")

if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)
shutil.copytree(best_adapter_path, FINAL_DIR)

results_log = {
    "timestamp": datetime.now().isoformat(),
    "smoke_test": True,
    "model": MODEL_NAME,
    "n_folds": N_FOLDS,
    "n_trials": N_TRIALS,
    "n_smoke_examples": N_SMOKE_EXAMPLES,
    "best_trial": best_trial.number,
    "best_params": best_trial.params,
    "best_mean_val_f1": best_trial.value,
    "best_fold": best_fold_idx,
    "all_trials": [
        {
            "trial": t.number,
            "params": t.params,
            "mean_val_f1": t.value,
            "fold_f1s": t.user_attrs.get("fold_f1s", []),
        }
        for t in study.trials
    ],
}
with open(os.path.join(RESULTS_DIR, "cv_results.json"), "w") as f:
    json.dump(results_log, f, indent=2, default=str)

print(f"\nSmoke results saved to {RESULTS_DIR}/cv_results.json")
print(f"Smoke adapter saved to {FINAL_DIR}")
print("\nSmoke test complete. If val_f1 appears above and cv_results.json looks correct, the full run is ready.")
