import os
import json
import gc
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ============================================================
# Configuration
# ============================================================
MODEL_NAME    = "Qwen/Qwen3.5-35B-A3B"
MAX_SEQ_LEN   = 512
DATA_CSV      = "data/training_set.csv"
MODEL_SLUG    = MODEL_NAME.split("/")[-1]
OUTPUT_DIR    = f"models/finetuned/{MODEL_SLUG}_saturation"
RESULTS_JSON  = "models/saturation_results_35B.json"
PLOT_PATH     = "models/saturation_curve_35B.png"

LORA_RANK     = 4
LORA_ALPHA    = 8
LEARNING_RATE = 0.00013199781284866172
NUM_EPOCHS    = 5
RANDOM_SEED   = 42

TRAIN_SIZES   = list(range(100, 701, 100))  # [100, 200, 300, 400, 500, 600, 700]

# ============================================================
# Helper functions (copied from 06.finetune.py)
# ============================================================
SYSTEM_PROMPT = (
    "You are a research assistant classifying job reviews.\n"
    "Determine whether the review contains any mention of work schedule.\n\n"
    "Mentions of schedule include:\n"
    "  - Working hours (long hours, short hours, specific shifts)\n"
    "  - Overtime (mandatory or voluntary)\n"
    "  - Flexibility or rigidity of hours\n"
    "  - Availability requirements (on-call, weekends, holidays)\n"
    "  - Stability or predictability of working hours\n\n"
    "  - Paid time off (PTO) policies, requests and flexibility around it"
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
        del inputs, out, new_tokens
    return f1_score(val_df["corrected_label"].tolist(), preds, zero_division=0)


# ============================================================
# Stratified cumulative sampling
# ============================================================
def build_cumulative_samples(train_df, sizes, seed):
    selected = []
    selected_set = set()
    result = {}
    for size in sizes:
        needed = size - len(selected)
        remaining = train_df[~train_df.index.isin(selected_set)]
        if needed >= len(remaining):
            new_idx = remaining.index.tolist()
        else:
            _, new_sample = train_test_split(
                remaining, test_size=needed,
                stratify=remaining["corrected_label"], random_state=seed,
            )
            new_idx = new_sample.index.tolist()
        selected.extend(new_idx)
        selected_set.update(new_idx)
        result[size] = list(selected)
    return result


# ============================================================
# Checkpointing
# ============================================================
def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["results"]
    return []


def save_results(path, results):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"results": results}, f, indent=2)
    os.replace(tmp, path)


# ============================================================
# Single training + evaluation step
# ============================================================
def train_and_evaluate(subset_df, eval_df, step_idx, output_dir):
    print(f"\n{'='*60}")
    print(f"  Step {step_idx + 1}/{len(TRAIN_SIZES)}  |  train_size={len(subset_df)}  |  eval_size={len(eval_df)}")
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
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
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
        "text": [
            format_example(row["doc"], row["corrected_label"])
            for _, row in subset_df.iterrows()
        ]
    })

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="no",
            save_strategy="no",
            output_dir=output_dir,
            seed=3407,
            dataloader_num_workers=0,
            dataset_num_proc=1,
        ),
        dataset_text_field="text",
    )

    trainer.train()

    f1 = compute_fold_f1(model, tokenizer, eval_df)
    print(f"\n  F1 on eval set: {f1:.4f}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

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

    return f1


# ============================================================
# Main
# ============================================================
print("Loading data...")
df = pd.read_csv(DATA_CSV)
train_df = df[df["set"] == 1].reset_index(drop=True)
eval_df  = df[df["set"] == 0].reset_index(drop=True)
print(f"Train pool: {len(train_df)}  |  Eval: {len(eval_df)}")

print("\nPre-computing cumulative sample sets...")
sample_map = build_cumulative_samples(train_df, TRAIN_SIZES, seed=RANDOM_SEED)
for size in TRAIN_SIZES:
    subset = train_df.loc[sample_map[size]]
    pos = int(subset["corrected_label"].sum())
    neg = int((subset["corrected_label"] == 0).sum())
    print(f"  size={size}: pos={pos}, neg={neg}, pos_ratio={pos/size:.3f}")

os.makedirs("models", exist_ok=True)
results = load_results(RESULTS_JSON)
completed_sizes = {r["train_size"] for r in results}

for step_idx, size in enumerate(TRAIN_SIZES):
    if size in completed_sizes:
        f1_done = next(r["f1"] for r in results if r["train_size"] == size)
        print(f"Skipping size={size} (already completed, F1={f1_done:.4f})")
        continue

    subset_df = train_df.loc[sample_map[size]].reset_index(drop=True)
    f1 = train_and_evaluate(subset_df, eval_df, step_idx, OUTPUT_DIR)
    results.append({"train_size": size, "f1": float(f1)})
    save_results(RESULTS_JSON, results)
    print(f"  Checkpoint saved: {RESULTS_JSON}")

# ============================================================
# Plot
# ============================================================
print("\nGenerating saturation curve plot...")
results_sorted = sorted(results, key=lambda r: r["train_size"])
sizes = [r["train_size"] for r in results_sorted]
f1s   = [r["f1"] for r in results_sorted]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sizes, f1s, "o-", linewidth=2, markersize=6, color="steelblue")
ax.set_xlabel("Training set size", fontsize=13)
ax.set_ylabel("F1 score (eval set)", fontsize=13)
ax.set_title("Data Saturation Curve — Qwen3.5-35B-A3B", fontsize=14)
ax.set_xticks(TRAIN_SIZES)
ax.set_xlim(left=0)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"Plot saved to {PLOT_PATH}")

print(f"\nAll done!")
print(f"Results: {RESULTS_JSON}")
print(f"Plot:    {PLOT_PATH}")
