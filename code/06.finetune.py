import os
import json
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
# Configuration
# ============================================================
MODEL_NAME  = "Qwen/Qwen3.5-9B"
MAX_SEQ_LEN = 2048
DATA_CSV    = "data/training_set.csv"
MODEL_SLUG  = MODEL_NAME.split("/")[-1]
RESULTS_DIR = f"models/cv_results/{MODEL_SLUG}"
FINAL_DIR   = f"models/finetuned/{MODEL_SLUG}"

N_FOLDS     = 5
N_TRIALS    = 20
PATIENCE    = 3              # Early stopping: stop after 3 evals without improvement

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
    """Format a single example in Qwen3.5 chat template."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{doc}<|im_end|>\n"
        f"<|im_start|>assistant\n{int(label)}<|im_end|>"
    )


def train_fold(train_dataset, val_dataset, config, fold_idx, output_dir):
    """Train a single fold with a given config. Returns best eval_loss."""

    print(f"\n{'='*60}")
    print(f"  Fold={fold_idx+1}/{N_FOLDS}  |  rank={config['lora_rank']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']}, lr={config['learning_rate']:.2e}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"{'='*60}\n")

    # Ensure previous trial/fold GPU memory is fully released before loading
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Load fresh model each run
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        #device_map="balanced",
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
            warmup_steps=10,
            num_train_epochs=5,
            learning_rate=config["learning_rate"],
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=25,
            eval_accumulation_steps=1,
            save_strategy="steps",
            save_steps=25,
            save_total_limit=2,
            output_dir=output_dir,
            seed=3407,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
        dataset_text_field="text",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()

    # Get best eval loss from training history
    eval_results = [
        log["eval_loss"]
        for log in trainer.state.log_history
        if "eval_loss" in log
    ]
    best_eval_loss = min(eval_results) if eval_results else float("inf")

    print(f"\n  Best eval_loss: {best_eval_loss:.4f}  |  stopped at epoch: {trainer.state.epoch:.2f}")

    # Save this fold's adapter
    fold_adapter_dir = os.path.join(output_dir, "best_adapter")
    os.makedirs(fold_adapter_dir, exist_ok=True)
    model.save_pretrained(fold_adapter_dir)
    tokenizer.save_pretrained(fold_adapter_dir)

    # Clean up GPU memory
    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return best_eval_loss


# ============================================================
# Main: Optuna Hyperparameter Search with K-Fold CV
# ============================================================
print("Loading data...")
df = pd.read_csv(DATA_CSV)
train_df_full = df[df["set"] == 1].reset_index(drop=True)
print(f"Total examples: {len(df)}  |  training set: {len(train_df_full)}")

os.makedirs(RESULTS_DIR, exist_ok=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=123)


def objective(trial):
    rank = trial.suggest_categorical("r", [4, 8, 16])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    lr = trial.suggest_float("lr", 1e-4, 3e-4, log=True)
    alpha_mode = trial.suggest_categorical("alpha_mode", ["equal_r", "double_r"])
    alpha = rank if alpha_mode == "equal_r" else 2 * rank

    config = {
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "learning_rate": lr,
    }

    print(f"\n{'*'*60}")
    print(f"  Trial {trial.number}  |  r={rank}, alpha={alpha}, dropout={dropout}, lr={lr:.2e}")
    print(f"{'*'*60}")

    fold_losses = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df_full, train_df_full["label"])):
        train_df = train_df_full.iloc[train_idx]
        val_df = train_df_full.iloc[val_idx]

        train_dataset = Dataset.from_dict({
            "text": [format_example(row["doc"], row["label"]) for _, row in train_df.iterrows()]
        })
        val_dataset = Dataset.from_dict({
            "text": [format_example(row["doc"], row["label"]) for _, row in val_df.iterrows()]
        })

        fold_output_dir = os.path.join(RESULTS_DIR, f"trial_{trial.number}", f"fold_{fold_idx}")

        best_loss = train_fold(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            fold_idx=fold_idx,
            output_dir=fold_output_dir,
        )
        fold_losses.append(best_loss)

    trial.set_user_attr("fold_losses", fold_losses)

    mean_loss = float(np.mean(fold_losses))
    print(f"\n  Trial {trial.number} done  |  mean_eval_loss={mean_loss:.4f}  |  folds: {[f'{l:.4f}' for l in fold_losses]}")
    return mean_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

# ============================================================
# Find best config and save its adapter
# ============================================================
print("\n" + "=" * 60)
print("OPTUNA SEARCH RESULTS SUMMARY")
print("=" * 60)

for t in sorted(study.trials, key=lambda x: x.value):
    print(f"  trial={t.number}  |  eval_loss={t.value:.4f}  |  params={t.params}")

best_trial = study.best_trial
best_fold_losses = best_trial.user_attrs["fold_losses"]
best_fold_idx = int(np.argmin(best_fold_losses))
best_adapter_path = os.path.join(RESULTS_DIR, f"trial_{best_trial.number}", f"fold_{best_fold_idx}", "best_adapter")

print(f"\nBest config: trial={best_trial.number}, params={best_trial.params} (mean_eval_loss={best_trial.value:.4f})")
print(f"Saving best adapter (trial={best_trial.number}, fold={best_fold_idx+1}) to {FINAL_DIR}")

if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)
shutil.copytree(best_adapter_path, FINAL_DIR)

# Save full results log
results_log = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "n_folds": N_FOLDS,
    "n_trials": N_TRIALS,
    "patience": PATIENCE,
    "best_trial": best_trial.number,
    "best_params": best_trial.params,
    "best_mean_eval_loss": best_trial.value,
    "best_fold": best_fold_idx,
    "all_trials": [
        {
            "trial": t.number,
            "params": t.params,
            "mean_eval_loss": t.value,
            "fold_losses": t.user_attrs.get("fold_losses", []),
        }
	        for t in study.trials
    ],
}
with open(os.path.join(RESULTS_DIR, "cv_results.json"), "w") as f:
    json.dump(results_log, f, indent=2, default=str)

print(f"\nFull results saved to {RESULTS_DIR}/cv_results.json")
print(f"Best adapter saved to {FINAL_DIR}")
print("Done!")
