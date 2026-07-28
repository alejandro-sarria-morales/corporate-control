"""
Single source of truth for the two-dimension prompt format, sequence budget, and
greedy two-digit decoding.

Imported by 11.finetune_multilabel.py (training) and 12.evaluate_multilabel.py
(inference) so the two cannot drift. They did drift once: 12 hand-copied the prompt
string and loaded the model at max_seq_length=2048 while 11 trained at 512, and the
512 silently truncated every training example before the answer digits -- the model
learned to reproduce half a system prompt and every generation came back malformed.

Deliberately torch-free at import time: `12 --preds_csv` (the GPT benchmark scoring
path) must keep working on a CPU-only box with no unsloth installed.

Runnable directly to validate the token budget without a GPU -- 11.finetune_multilabel.py
imports unsloth at module level, so it cannot run on a login node:

    python code/multilabel_prompt.py [path/to/labelled.csv]

Doing that on a login node also warms the tokenizer into HF_HOME, which matters because
DCC compute nodes have no internet.
"""

import os

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
MODEL_SLUG = MODEL_NAME.split("/")[-1]

# Must hold the system prompt (~1030 tokens) + the review + the two answer digits.
# 11.finetune_multilabel.py runs a preflight that measures this against the real
# tokenizer and refuses to start if it is too small.
MAX_SEQ_LEN = 1536

PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts", "multilabel_system_prompt.txt"
)

with open(PROMPT_FILE, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()


def format_prompt(doc):
    """Inference prompt: everything up to (not including) the answer."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{doc}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_answer(sched, ctrl):
    """The supervised completion: two digits plus the terminator the model must emit."""
    return f"{int(sched)}{int(ctrl)}<|im_end|>"


def format_example(doc, sched, ctrl):
    """Full training sequence. Always exactly format_prompt + format_answer."""
    return format_prompt(doc) + format_answer(sched, ctrl)


def strip_reasoning(text):
    """Drop a leading reasoning block if one ever appears.

    Belt-and-braces only: nothing in this repo calls apply_chat_template, so there is
    no code path that auto-injects a <think> block, and 07.evaluate_qwen.py produced
    clean bare digits on this same architecture.
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def im_end_id(tokenizer):
    """Token id of <|im_end|>, falling back to eos if the tokenizer lacks it.

    The tokenizer remaps eos_token_id to 248046, but the training target terminates
    with <|im_end|>, so generation must stop on that instead.
    """
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tid is None or tid < 0:
        return tokenizer.eos_token_id
    return tid


ANSWER_MARGIN = 8  # room for <|im_end|>, pad, and chat-template drift


def preflight_token_budget(df, max_seq_len=None):
    """Fail in seconds if max_seq_len cannot hold prompt + review + answer.

    This is the check that would have caught the v1 failure before it spent four days:
    MAX_SEQ_LEN=512 against a ~1030-token system prompt truncated every training
    sequence before the answer digits, so the model was never supervised on the answer
    and every generation came back malformed.

    Tokenizer only -- no GPU, no unsloth, ~3s.
    """
    from transformers import AutoTokenizer  # local: keeps this module torch-free to import

    import numpy as np

    max_seq_len = MAX_SEQ_LEN if max_seq_len is None else max_seq_len
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    n_sys = len(tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"])

    lens = np.asarray([
        len(tok(format_example(r["review_text"], r["schedule_related"],
                               r["job_control_related"]),
                add_special_tokens=False)["input_ids"])
        for _, r in df.iterrows()
    ])
    need = int(lens.max()) + ANSWER_MARGIN

    print("\n" + "=" * 60)
    print("PREFLIGHT: token budget")
    print(f"  system prompt                : {n_sys}")
    print(f"  full example min/med/max     : {lens.min()} / {int(np.median(lens))} / {lens.max()}")
    print(f"  required (max + {ANSWER_MARGIN} margin)   : {need}")
    print(f"  MAX_SEQ_LEN                  : {max_seq_len}")
    print("=" * 60)

    if need > max_seq_len:
        raise SystemExit(
            f"FATAL: MAX_SEQ_LEN={max_seq_len} truncates training examples.\n"
            f"  Longest example needs {need} tokens; the system prompt alone is {n_sys}.\n"
            f"  The answer digits would be cut off, the model would train on a constant\n"
            f"  prompt prefix, and every generation would be malformed.\n"
            f"  Set MAX_SEQ_LEN >= {((need + 63) // 64) * 64} in code/multilabel_prompt.py."
        )

    # Direct check: does the answer actually survive right-truncation?
    w = df.iloc[int(np.argmax(lens))]
    ids = tok(format_example(w["review_text"], w["schedule_related"], w["job_control_related"]),
              add_special_tokens=False)["input_ids"][:max_seq_len]
    tail = tok.decode(ids[-8:])
    expected = f"{int(w['schedule_related'])}{int(w['job_control_related'])}"
    if expected not in tail:
        raise SystemExit(
            f"FATAL: answer {expected!r} missing from the last 8 tokens after truncation "
            f"at MAX_SEQ_LEN={max_seq_len}; tail decoded as {tail!r}"
        )
    print(f"  answer survives truncation   : OK (tail = {tail!r})\n")


def generate_two_digit(model, tokenizer, doc, max_new_tokens=8):
    """Greedy decode one review; return the raw generated text (reasoning stripped)."""
    import torch  # local import keeps this module importable without torch

    inputs = tokenizer(format_prompt(doc), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=im_end_id(tokenizer),
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    del inputs, out
    return strip_reasoning(text)


if __name__ == "__main__":
    # Login-node validation: checks the token budget and warms the tokenizer cache.
    import sys
    import pandas as pd

    csv = sys.argv[1] if len(sys.argv) > 1 else "data/trainingfinal/labelled.csv"
    print(f"Checking token budget for {csv} (MAX_SEQ_LEN={MAX_SEQ_LEN})")
    preflight_token_budget(pd.read_csv(csv))
    print("Preflight OK.")
