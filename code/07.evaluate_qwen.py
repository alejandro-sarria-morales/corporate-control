import os
import torch
import pandas as pd
from sklearn.metrics import classification_report

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from unsloth import FastModel

# ============================================================
# Configuration
# ============================================================
MODEL_NAME  = "Qwen/Qwen3.5-35B-A3B"
ADAPTER_DIR = f"models/finetuned/{MODEL_NAME.split('/')[-1]}"
DATA_CSV    = "data/training_set.csv"

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

# ============================================================
# Load held-out evaluation set
# ============================================================
print("Loading evaluation data...")
df = pd.read_csv(DATA_CSV)
val_df = df[df["set"] == 0].reset_index(drop=True)
print(f"Evaluation examples: {len(val_df)}")

# ============================================================
# Load model + adapter
# ============================================================
print("Loading model...")
model, tokenizer = FastModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_16bit=False,
    full_finetuning=False,
    device_map="balanced"
)

#model = model.to("cuda")

if hasattr(tokenizer, "tokenizer"):
    tokenizer = tokenizer.tokenizer

# ============================================================
# Run predictions
# ============================================================
print("Running predictions...")
preds, labels = [], []
for i, (_, row) in enumerate(val_df.iterrows()):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{row['doc']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    pred = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    preds.append(pred)
    labels.append(str(int(row["label"])))

    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(val_df)} done")

# ============================================================
# Save validation df with predictions.
# ============================================================
val_df = val_df.copy()
val_df["preds"] = preds
val_df.to_csv(f"data/val_preds_{MODEL_NAME.split('/')[-1]}.csv", index=False)
print("Saved validation predictions.")

# ============================================================
# Results
# ============================================================
print("\n" + "=" * 50)
print("Classification Report:")
print("=" * 50)
print(classification_report(labels, preds, target_names=["No schedule (0)", "Schedule (1)"]))

# Show any unexpected predictions (not 0 or 1)
unexpected = [(p, l) for p, l in zip(preds, labels) if p not in ("0", "1")]
if unexpected:
    print(f"\nWarning: {len(unexpected)} predictions were not 0 or 1:")
    for p, l in unexpected[:10]:
        print(f"  Predicted: '{p}', True label: {l}")
