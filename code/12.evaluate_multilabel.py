"""
Evaluate the two-dimension classifier on the frozen held-out test set (set==0).

Two modes:

  (1) Fine-tuned model (default):
        python code/12.evaluate_multilabel.py
      Loads the LoRA adapter, generates two-digit predictions on the test set, prints
      the full multi-label report, and writes data/trainingfinal/val_preds_<slug>.csv.

  (2) Score an existing predictions file (used for the GPT benchmark, so it is scored
      with the *identical* metric code and on the *identical* test rows):
        python code/12.evaluate_multilabel.py --preds_csv data/trainingfinal/gpt_preds.csv --name gpt-5.4
      The preds CSV must have an `id` column plus either `sched_pred`+`ctrl_pred`
      columns or a single two-digit `pred` column.
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_multilabel import parse_two_digit, multilabel_report
from multilabel_prompt import MODEL_SLUG, MAX_SEQ_LEN, generate_two_digit

ADAPTER_DIR = f"models/finetuned/{MODEL_SLUG}_multilabel"
DATA_CSV    = "data/trainingfinal/labelled.csv"
TARGET_F1   = 0.95


def load_test_set():
    df = pd.read_csv(DATA_CSV)
    test = df[df["set"] == 0].reset_index(drop=True)
    print(f"Held-out test rows: {len(test)}  "
          f"(schedule pos={int(test['schedule_related'].sum())}, "
          f"job_control pos={int(test['job_control_related'].sum())})")
    return test


def score_from_preds_csv(path, name):
    """Score a predictions file against the frozen test gold labels."""
    test = load_test_set()
    preds = pd.read_csv(path)
    if "id" not in preds.columns:
        raise ValueError("preds CSV must contain an 'id' column to join on gold labels")

    if {"sched_pred", "ctrl_pred"}.issubset(preds.columns):
        preds = preds[["id", "sched_pred", "ctrl_pred"]].copy()
        n_malformed = 0
    elif "pred" in preds.columns:
        parsed = preds["pred"].map(parse_two_digit)
        preds = preds.assign(
            sched_pred=[p[0] for p in parsed],
            ctrl_pred=[p[1] for p in parsed],
        )[["id", "sched_pred", "ctrl_pred"]]
        n_malformed = int(sum(not p[2] for p in parsed))
    else:
        raise ValueError("preds CSV needs 'sched_pred'+'ctrl_pred' or a 'pred' column")

    merged = test.merge(preds, on="id", how="left", validate="one_to_one")
    missing = merged["sched_pred"].isna().sum()
    if missing:
        raise ValueError(f"{missing} test rows have no prediction in {path}")

    multilabel_report(
        merged["schedule_related"], merged["job_control_related"],
        merged["sched_pred"].astype(int), merged["ctrl_pred"].astype(int),
        name=name, target_f1=TARGET_F1, n_malformed=n_malformed,
    )


def score_finetuned_model():
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from unsloth import FastModel

    test = load_test_set()

    print("Loading model + adapter...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=ADAPTER_DIR,
        max_seq_length=MAX_SEQ_LEN,  # match training; see multilabel_prompt.py
        load_in_4bit=True,
        load_in_16bit=False,
        full_finetuning=False,
        device_map="balanced",
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    print("Running predictions...")
    raw, sp, cp, malformed = [], [], [], 0
    for i, (_, row) in enumerate(test.iterrows()):
        text = generate_two_digit(model, tokenizer, row["review_text"], max_new_tokens=8)
        s, c, ok = parse_two_digit(text)
        malformed += (not ok)
        raw.append(text); sp.append(s); cp.append(c)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(test)} done")

    out_df = test[["id", "review_text", "type", "schedule_related", "job_control_related"]].copy()
    out_df["sched_pred"] = sp
    out_df["ctrl_pred"] = cp
    out_df["raw_output"] = raw
    out_path = f"data/trainingfinal/val_preds_{MODEL_SLUG}_multilabel.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    multilabel_report(
        test["schedule_related"], test["job_control_related"], sp, cp,
        name=f"{MODEL_SLUG} (finetuned)", target_f1=TARGET_F1, n_malformed=malformed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_csv", default=None,
                    help="Score an existing predictions CSV instead of loading the model.")
    ap.add_argument("--name", default=None, help="Label for the report header.")
    args = ap.parse_args()

    if args.preds_csv:
        score_from_preds_csv(args.preds_csv, args.name or os.path.basename(args.preds_csv))
    else:
        score_finetuned_model()


if __name__ == "__main__":
    main()
