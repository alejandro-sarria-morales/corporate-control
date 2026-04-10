import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


#==========================
# config
#==========================

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
MAX_SEQ_LEN = 2048
DATA_CSV = "data/training_set.csv"
OUTPUT_DIR = "models/checkpoints"
FINAL_DIR = "models/final"

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

# ============================================================
# Step 1: Prepare dataset
# ============================================================
def format_example(doc, label):
    """Format a single example in Qwen3.5 chat template."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{doc}<|im_end|>\n"
        f"<|im_start|>assistant\n{int(label)}<|im_end|>"
    )

print("Preparing dataset...")
df = pd.read_csv(DATA_CSV)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=123, stratify=df["label"])

train_dataset = Dataset.from_dict({"text": [format_example(r["doc"], r["label"]) for _, r in train_df.iterrows()]})
val_dataset   = Dataset.from_dict({"text": [format_example(r["doc"], r["label"]) for _, r in val_df.iterrows()]})

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ============================================================
# Step 2: Load model in bf16 with LoRA
# ============================================================
print("Loading model...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,       # QLoRA not recommended for Qwen3.5
    load_in_16bit=True,       # bf16 LoRA
    full_finetuning=False,
    device_map="balanced"
)

# If FastModel returns a processor (Qwen3.5 is a VLM), extract tokenizer
if hasattr(tokenizer, "tokenizer"):
    tokenizer = tokenizer.tokenizer

model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,            # alpha == r per Unsloth recommendation
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

# ============================================================
# Step 3: Train
# ============================================================
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,    # Effective batch size = 8
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        output_dir=OUTPUT_DIR,
        seed=3407,
        dataloader_num_workers=0,         # Prevent fork deadlock
        dataset_num_proc=1,               # Prevent fork deadlock
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    ),
    dataset_text_field="text",
)

trainer.train()

# ============================================================
# Step 4: Save final adapter
# ============================================================
os.makedirs(FINAL_DIR, exist_ok=True)
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"LoRA adapter saved to {FINAL_DIR}")
