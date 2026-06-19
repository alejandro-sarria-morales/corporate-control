import os
import torch
import pandas as pd

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from unsloth import FastModel

# ============================================================
# Configuration
# ============================================================
MODEL_NAME       = "Qwen/Qwen3.5-35B-A3B"
ADAPTER_DIR      = f"models/finetuned/{MODEL_NAME.split('/')[-1]}"
INPUT_CSV        = "data/glassdoor_reviews_clean.csv"
OUTPUT_CSV       = "data/glassdoor_labelled.csv"
CHECKPOINT_EVERY = 500

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


def classify(text, model, tokenizer):
    """Return 1, 0, or None if text is too short to classify."""
    if not isinstance(text, str) or len(text.split()) < 3:
        return None
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    return 1 if pred.startswith("1") else 0


# ============================================================
# Load data
# ============================================================
print("Loading glassdoor data...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"  {len(df):,} rows loaded")

# ============================================================
# Checkpoint / resume
# ============================================================
if os.path.exists(OUTPUT_CSV):
    print(f"Resuming from existing output: {OUTPUT_CSV}")
    done = pd.read_csv(OUTPUT_CSV, usecols=["reviewID", "label_pros", "label_cons"],
                       low_memory=False)
    # Drop duplicate reviewIDs to prevent row explosion on merge
    done = done.drop_duplicates(subset="reviewID", keep="first")
    # Drop rows where both labels are NaN — these were never classified (old checkpoint bug
    # wrote the whole df including unprocessed rows; NaN from too-short text is kept because
    # at least one of the two fields will typically differ, and we accept re-running the rare
    # case where both fields are too short)
    done = done[~(done["label_pros"].isna() & done["label_cons"].isna())]
    already_done = set(done["reviewID"].tolist())
    # Merge existing labels back into df
    df = df.merge(done, on="reviewID", how="left")
    n_done = df["reviewID"].isin(already_done).sum()
    print(f"  {n_done:,} rows already labelled, {(~df['reviewID'].isin(already_done)).sum():,} remaining")
else:
    df["label_pros"] = None
    df["label_cons"] = None
    already_done = set()

# ============================================================
# Load model
# ============================================================
print("Loading model...")
model, tokenizer = FastModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_16bit=False,
    full_finetuning=False,
    device_map="balanced",
)

if hasattr(tokenizer, "tokenizer"):
    tokenizer = tokenizer.tokenizer

# ============================================================
# Inference loop
# ============================================================
todo_idx = df.index[~df["reviewID"].isin(already_done)].tolist()
n_todo = len(todo_idx)
print(f"Running inference on {n_todo:,} rows...")

for step, idx in enumerate(todo_idx):
    row = df.loc[idx]
    df.at[idx, "label_pros"] = classify(row["review_pros"], model, tokenizer)
    df.at[idx, "label_cons"] = classify(row["review_cons"], model, tokenizer)
    already_done.add(row["reviewID"])  # mark as processed so checkpoint includes it

    if (step + 1) % 100 == 0:
        print(f"  {step + 1:,}/{n_todo:,} ({(step + 1) / n_todo * 100:.1f}%)")

    if (step + 1) % CHECKPOINT_EVERY == 0:
        # Only write processed rows — avoids NaN-polluting the output with unprocessed rows
        df[df["reviewID"].isin(already_done)].to_csv(OUTPUT_CSV, index=False)
        print(f"  [checkpoint saved]")

# ============================================================
# Final save
# ============================================================
saved = df[df["reviewID"].isin(already_done)]
saved.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. Output saved to {OUTPUT_CSV}")
print(f"  label_pros distribution:\n{saved['label_pros'].value_counts(dropna=False)}")
print(f"  label_cons distribution:\n{saved['label_cons'].value_counts(dropna=False)}")
