import os
import torch
import pandas as pd

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel

# ============================================================
# Configuration
# ============================================================
MODEL_NAME       = "Qwen/Qwen3.5-35B-A3B"
ADAPTER_DIR      = f"models/finetuned/{MODEL_NAME.split('/')[-1]}"
INPUT_CSV        = "data/glassdoor_reviews_clean.csv"
OUTPUT_CSV       = "data/glassdoor_labelled.csv"
CHECKPOINT_EVERY = 500
ROW_BATCH_SIZE   = 8

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


def classify_batch(texts, model, tokenizer):
    """Return a list of 1, 0, or None (for texts too short to classify), in order."""
    results = [None] * len(texts)
    keep_idx = [i for i, t in enumerate(texts) if isinstance(t, str) and len(t.split()) >= 3]
    if not keep_idx:
        return results

    prompts = [
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{texts[i]}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for i in keep_idx
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    prompt_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
    for i, pred in zip(keep_idx, decoded):
        results[i] = 1 if pred.strip().startswith("1") else 0
    return results


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

# Left padding required so batched prompts share a common length for slicing
# the generated continuation off the end of each sequence.
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# Inference loop
# ============================================================
todo_idx = df.index[~df["reviewID"].isin(already_done)].tolist()
n_todo = len(todo_idx)
print(f"Running inference on {n_todo:,} rows...")

rows_done = 0
last_checkpoint = 0
for chunk_start in range(0, n_todo, ROW_BATCH_SIZE):
    chunk_idx = todo_idx[chunk_start:chunk_start + ROW_BATCH_SIZE]
    chunk = df.loc[chunk_idx]

    pros_preds = classify_batch(chunk["review_pros"].tolist(), model, tokenizer)
    cons_preds = classify_batch(chunk["review_cons"].tolist(), model, tokenizer)

    for idx, pros_pred, cons_pred in zip(chunk_idx, pros_preds, cons_preds):
        df.at[idx, "label_pros"] = pros_pred
        df.at[idx, "label_cons"] = cons_pred
        already_done.add(df.at[idx, "reviewID"])

    rows_done += len(chunk_idx)
    torch.cuda.empty_cache()
    print(f"  {rows_done:,}/{n_todo:,} ({rows_done / n_todo * 100:.1f}%)")

    if rows_done - last_checkpoint >= CHECKPOINT_EVERY:
        # Only write processed rows — avoids NaN-polluting the output with unprocessed rows
        df[df["reviewID"].isin(already_done)].to_csv(OUTPUT_CSV, index=False)
        print(f"  [checkpoint saved]")
        last_checkpoint = rows_done

# ============================================================
# Final save
# ============================================================
saved = df[df["reviewID"].isin(already_done)]
saved.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. Output saved to {OUTPUT_CSV}")
print(f"  label_pros distribution:\n{saved['label_pros'].value_counts(dropna=False)}")
print(f"  label_cons distribution:\n{saved['label_cons'].value_counts(dropna=False)}")
