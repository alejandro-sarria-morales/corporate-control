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
    load_in_4bit=False,
    load_in_16bit=True,
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
