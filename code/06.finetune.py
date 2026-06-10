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
# Configuration
# ============================================================
MODEL_NAME  = "Qwen/Qwen3.5-35B-A3B"
MAX_SEQ_LEN = 512
DATA_CSV    = "data/training_set.csv"
MODEL_SLUG  = MODEL_NAME.split("/")[-1]
RESULTS_DIR = f"models/cv_results/{MODEL_SLUG}"
FINAL_DIR   = f"models/finetuned/{MODEL_SLUG}"

N_FOLDS     = 5
N_TRIALS    = 5
PATIENCE    = 3              # Early stopping: stop after 3 evals without improvement

SYSTEM_PROMPT = (
    "You are a research assistant classifying job reviews.\n"
    "Label 1 if the review contains content about work schedule, workload, or job flexibility. Label 0 otherwise.\n\n"
    "Content that earns label 1:\n"
    "  - Hours of work: long, short, specified, variable, night shifts, overtime\n"
    "  - Schedule control: flexible or rigid schedules, on-call, required weekends/holidays, breaks\n"
    "  - Schedule predictability: last-minute changes, advance notice\n"
    "  - Paid time off: PTO, vacation, sick leave\n"
    "  - Work location: remote, work-from-home, hybrid arrangements\n"
    "  - Workload intensity: overworked, understaffed, or too little to do\n"
    "  - Job flexibility or autonomy in any context\n"
    "  - Explicit work-life balance or work-life conflict mentions\n\n"
    "These look similar but earn label 0:\n"
    "  - Vague time references not tied to schedule: 'time away from home', 'time spent in meetings'\n"
    "  - Stress without schedule or workload language: 'stressful job', 'a lot of work'\n"
    "  - Coworker quality, culture, career growth, or pay without schedule or workload content\n\n"
    "Label 1 examples:\n"
    "  'long hours and a lot of work'\n"
    "  'they offer flexible hours and the staff is very nice'\n"
    "  'not enough hours a week regardless if you are willing to work more'\n"
    "  'on call 24/7 and they change your schedule last minute'\n"
    "  'no paid sick days or vacation time'\n"
    "  'no option to work from home even when you could do the job remotely'\n"
    "  'good work-life balance in the company'\n"
    "  'you are always understaffed and have a lot of work to do'\n"
    "  'no overtime, no 100% set career path'\n"
    "  'great flexibility, enjoyable work environment'\n\n"
    "Label 0 examples:\n"
    "  'tight-knit office, everyone became friends, free lunch on thursdays'\n"
    "  'career movement, leadership not diversed enough'\n"
    "  'people are boring and the workplace conversations are limited'\n"
    "  'too much time away from home'\n"
    "  'can expect a lot of time going into meetings'\n"
    "  'stressful some days; a lot of work'\n\n"
    "Reply with exactly one character: 1 or 0. No explanation or punctuation."
)


def format_example(doc, label):
    """Format a single example in Qwen3.5 chat template."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{doc}<|im_end|>\n"
        f"<|im_start|>assistant\n{int(label)}<|im_end|>"
    )


def format_prompt(doc):
    """Format an inference prompt without the assistant answer."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{doc}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def compute_fold_f1(model, tokenizer, val_df):
    """Run greedy inference on val_df and return binary F1 (positive class=1)."""
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
        del inputs, out, new_tokens
    return f1_score(val_df["corrected_label"].tolist(), preds, zero_division=0)


def train_fold(train_dataset, val_dataset, config, fold_idx, output_dir, val_df):
    """Train a single fold with a given config. Returns val F1 on the best checkpoint."""

    print(f"\n{'='*60}")
    print(f"  Fold={fold_idx+1}/{N_FOLDS}  |  rank={config['lora_rank']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']}, lr={config['learning_rate']:.2e}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"{'='*60}\n")

    # Ensure previous trial/fold GPU memory is fully released before loading
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    cache_dir = os.path.join(os.getcwd(), "unsloth_compiled_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # Load fresh model each run
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
        lora_dropout=0.0,  # MoE ParamWrapper layers (expert weights) require dropout=0
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
	    prediction_loss_only=True
        ),
        dataset_text_field="text",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()

    # Compute F1 on the val fold using the best-loss checkpoint (loaded automatically)
    val_f1 = compute_fold_f1(model, tokenizer, val_df)
    print(f"\n  Val F1: {val_f1:.4f}  |  stopped at epoch: {trainer.state.epoch:.2f}")

    # Save this fold's adapter
    fold_adapter_dir = os.path.join(output_dir, "best_adapter")
    os.makedirs(fold_adapter_dir, exist_ok=True)
    model.save_pretrained(fold_adapter_dir)
    tokenizer.save_pretrained(fold_adapter_dir)

    # Sever internal trainer references before deletion so GC can free GPU tensors
    trainer.model = None
    trainer.optimizer = None
    trainer.lr_scheduler = None
    for cb in trainer.callback_handler.callbacks:
        if hasattr(cb, "model"):
            cb.model = None
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    return val_f1


def train_final(train_df, config):
    """Retrain on the full training set with the best config and save to FINAL_DIR."""
    print(f"\n{'='*60}")
    print(f"  FINAL TRAINING on full dataset ({len(train_df)} examples)")
    print(f"  r={config['lora_rank']}, alpha={config['lora_alpha']}, lr={config['learning_rate']:.2e}")
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
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=123,
        max_seq_length=MAX_SEQ_LEN,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    train_dataset = Dataset.from_dict({
        "text": [format_example(row["doc"], row["corrected_label"]) for _, row in train_df.iterrows()]
    })

    os.makedirs(FINAL_DIR, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
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

    trainer.model = None
    trainer.optimizer = None
    trainer.lr_scheduler = None
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


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
    rank = trial.suggest_categorical("r", [4, 8, 16, 32])
    lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
    alpha_mode = trial.suggest_categorical("alpha_mode", ["r", "2r"])
    alpha = rank if alpha_mode == "r" else 2 * rank

    config = {
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,  # MoE ParamWrapper requires dropout=0
        "learning_rate": lr,
    }

    print(f"\n{'*'*60}")
    print(f"  Trial {trial.number}  |  r={rank}, alpha={alpha}, lr={lr:.2e}")
    print(f"{'*'*60}")

    fold_f1s = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df_full, train_df_full["label"])):
        train_df = train_df_full.iloc[train_idx]
        val_df = train_df_full.iloc[val_idx]

        train_dataset = Dataset.from_dict({
            "text": [format_example(row["doc"], row["corrected_label"]) for _, row in train_df.iterrows()]
        })
        val_dataset = Dataset.from_dict({
            "text": [format_example(row["doc"], row["corrected_label"]) for _, row in val_df.iterrows()]
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

    # Clean up: keep only the best fold's adapter, delete the rest
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
study.optimize(objective, n_trials=N_TRIALS, catch=(torch.cuda.OutOfMemoryError, NotImplementedError, RuntimeError, ValueError))

# ============================================================
# Find best config and save its adapter
# ============================================================
print("\n" + "=" * 60)
print("OPTUNA SEARCH RESULTS SUMMARY")
print("=" * 60)

for t in sorted([t for t in study.trials if t.value is not None], key=lambda x: x.value, reverse=True):
    print(f"  trial={t.number}  |  mean_val_f1={t.value:.4f}  |  params={t.params}")

best_trial = study.best_trial

print(f"\nBest config: trial={best_trial.number}, params={best_trial.params} (mean_val_f1={best_trial.value:.4f})")
print("Retraining on full training set with best config...")

best_rank = best_trial.params["r"]
best_alpha_mode = best_trial.params["alpha_mode"]
best_config = {
    "lora_rank": best_rank,
    "lora_alpha": best_rank if best_alpha_mode == "r" else 2 * best_rank,
    "lora_dropout": 0.0,
    "learning_rate": best_trial.params["lr"],
}

if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)

train_final(train_df_full, best_config)

# Save full results log
results_log = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "n_folds": N_FOLDS,
    "n_trials": N_TRIALS,
    "patience": PATIENCE,
    "best_trial": best_trial.number,
    "best_params": best_trial.params,
    "best_mean_val_f1": best_trial.value,
    "final_model": "retrained on full training set with best_params",
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

print(f"\nFull results saved to {RESULTS_DIR}/cv_results.json")
print(f"Final model saved to {FINAL_DIR}")
print("Done!")
