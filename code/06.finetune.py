import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import shutil

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
RESULTS_DIR = "models/cv_results"
FINAL_DIR   = "models/final"

N_FOLDS     = 3
LORA_RANKS  = [4, 8, 16]
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
    "Here are two examples of a review talking about schedule:\n"
    "  1. 'long hours and a lot of work'\n"
    "  2. 'they offer flexible hours and the staff is very nice.'\n\n"
    "Here are three examples of a review not talking about schedule:\n"
    "  1. 'i have been working at alta resources for sunrun part-time for more than a year "
    "the brea office is a tight family, everyone became friends and free lunch on thursdays.'\n"
    "  2. 'people are boring and the workplace conversations are limited. some people are not "
    "very good at their jobs and they\\'re allowed to stay way too long.'\n\n"
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


def train_fold(train_dataset, val_dataset, lora_rank, fold_idx, output_dir):
    """Train a single fold with a given LoRA rank. Returns best eval_loss."""

    print(f"\n{'='*60}")
    print(f"  Rank={lora_rank}, Fold={fold_idx+1}/{N_FOLDS}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"{'='*60}\n")

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
        r=lora_rank,
        lora_alpha=lora_rank,       # alpha == r per Unsloth recommendation
        lora_dropout=0,
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
            num_train_epochs=10,          # High ceiling — early stopping will cut it short
            learning_rate=2e-4,
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=25,                # Eval more frequently for early stopping
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

    print(f"\n  Best eval_loss for rank={lora_rank}, fold={fold_idx+1}: {best_eval_loss:.4f}")
    print(f"  Stopped at epoch: {trainer.state.epoch:.2f}")

    # Save this fold's adapter
    fold_adapter_dir = os.path.join(output_dir, "best_adapter")
    os.makedirs(fold_adapter_dir, exist_ok=True)
    model.save_pretrained(fold_adapter_dir)
    tokenizer.save_pretrained(fold_adapter_dir)

    # Clean up GPU memory
    del model, tokenizer, trainer
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return best_eval_loss


# ============================================================
# Main: Grid Search with K-Fold CV
# ============================================================
print("Loading data...")
df = pd.read_csv(DATA_CSV)
print(f"Total examples: {len(df)}")

os.makedirs(RESULTS_DIR, exist_ok=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=123)
all_results = []

for lora_rank in LORA_RANKS:
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = Dataset.from_dict({
            "text": [format_example(r["doc"], r["label"]) for _, r in train_df.iterrows()]
        })
        val_dataset = Dataset.from_dict({
            "text": [format_example(r["doc"], r["label"]) for _, r in val_df.iterrows()]
        })

        fold_output_dir = os.path.join(RESULTS_DIR, f"rank_{lora_rank}", f"fold_{fold_idx}")

        best_loss = train_fold(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            lora_rank=lora_rank,
            fold_idx=fold_idx,
            output_dir=fold_output_dir,
        )
        fold_losses.append(best_loss)

    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)

    result = {
        "lora_rank": lora_rank,
        "fold_losses": fold_losses,
        "mean_eval_loss": mean_loss,
        "std_eval_loss": std_loss,
    }
    all_results.append(result)

    print(f"\n{'*'*60}")
    print(f"  Rank={lora_rank}: mean_eval_loss={mean_loss:.4f} +/- {std_loss:.4f}")
    print(f"  Per-fold losses: {[f'{l:.4f}' for l in fold_losses]}")
    print(f"{'*'*60}\n")

# ============================================================
# Find best config and save its adapter
# ============================================================
print("\n" + "=" * 60)
print("GRID SEARCH RESULTS SUMMARY")
print("=" * 60)

for r in all_results:
    print(f"  rank={r['lora_rank']:>3d}  |  eval_loss = {r['mean_eval_loss']:.4f} +/- {r['std_eval_loss']:.4f}  |  folds: {[f'{l:.4f}' for l in r['fold_losses']]}")

best_result = min(all_results, key=lambda x: x["mean_eval_loss"])
best_rank = best_result["lora_rank"]

print(f"\nBest config: rank={best_rank} (mean_eval_loss={best_result['mean_eval_loss']:.4f})")

# Find best fold for the best rank
best_fold_idx = np.argmin(best_result["fold_losses"])
best_adapter_path = os.path.join(RESULTS_DIR, f"rank_{best_rank}", f"fold_{best_fold_idx}", "best_adapter")

# Copy best adapter to final directory
print(f"Saving best adapter (rank={best_rank}, fold={best_fold_idx+1}) to {FINAL_DIR}")
os.makedirs(FINAL_DIR, exist_ok=True)

if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)
shutil.copytree(best_adapter_path, FINAL_DIR)

# Save full results log
results_log = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "n_folds": N_FOLDS,
    "lora_ranks_tested": LORA_RANKS,
    "patience": PATIENCE,
    "results": all_results,
    "best_rank": best_rank,
    "best_fold": int(best_fold_idx),
    "best_mean_eval_loss": best_result["mean_eval_loss"],
}
with open(os.path.join(RESULTS_DIR, "cv_results.json"), "w") as f:
    json.dump(results_log, f, indent=2, default=str)

print(f"\nFull results saved to {RESULTS_DIR}/cv_results.json")
print(f"Best adapter saved to {FINAL_DIR}")
print("Done!")
