"""
Prepare the frozen multi-label training/eval dataset.

Reads the gold-coded workbook and produces data/trainingfinal/labelled.csv, the
single artifact consumed by 11.finetune_multilabel.py, 12.evaluate_multilabel.py,
and 13.GPT_multilabel_benchmark.R.

What it guarantees:
  - schedule_related / job_control_related are clean 0/1 ints.
  - not_related is the exact complement of the other two (asserted).
  - Duplicate review texts never straddle the train/test boundary (dedup-before-split),
    so there is no leakage between sets.
  - The few-shot examples embedded in code/prompts/multilabel_system_prompt.txt are
    forced into the TRAIN split. Because those examples appear in the system prompt for
    EVERY prediction, letting one fall into the test set would leak its gold answer.
  - A fixed, stratified ~75/25 split on the 4-way state key (2*schedule + job_control),
    written to the `set` column (1 = train, 0 = held-out test) to match the existing
    binary pipeline convention.
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SRC_XLSX   = "data/trainingfinal/dataset_three_categories.xlsx"
SHEET      = "three_categories"
OUT_CSV    = "data/trainingfinal/labelled.csv"
TEST_FRAC  = 0.25
RANDOM_SEED = 42

# ids of the few-shot examples hardcoded in the system prompt; must stay in TRAIN
FEWSHOT_IDS = {
    45502, 78714, 30487, 13051, 98957,      # schedule-only (10)
    55488, 31910, 77436, 87495, 97736,      # job-control-only (01)
    69473, 78486, 89252, 51994,             # both (11)
    51350, 9892, 21959, 55390, 70129,       # neither (00)
}


def norm(text) -> str:
    """Whitespace/case-normalized key for duplicate detection."""
    return " ".join(str(text).split()).lower()


def main() -> None:
    df = pd.read_excel(SRC_XLSX, sheet_name=SHEET)
    print(f"Loaded {len(df)} rows from {SRC_XLSX}")

    # --- clean labels -------------------------------------------------
    for col in ("schedule_related", "job_control_related", "not_related"):
        if df[col].isna().any():
            raise ValueError(f"NA values in label column {col!r}")
        df[col] = df[col].astype(int)
        bad = set(df[col].unique()) - {0, 1}
        if bad:
            raise ValueError(f"Non-binary values {bad} in {col!r}")

    # --- integrity: not_related is the complement of the other two ----
    either = ((df["schedule_related"] == 1) | (df["job_control_related"] == 1)).astype(int)
    violations = (df["not_related"] != (1 - either)).sum()
    if violations:
        raise ValueError(f"{violations} rows where not_related != 1-(schedule|job_control)")
    print("Integrity check passed: not_related == 1 - (schedule | job_control)")

    # --- 4-way stratification key -------------------------------------
    df["strat"] = 2 * df["schedule_related"] + df["job_control_related"]
    state_name = {0: "none(00)", 1: "jobctrl(01)", 2: "sched(10)", 3: "both(11)"}
    print("\nState distribution:")
    for k, n in df["strat"].value_counts().sort_index().items():
        print(f"  {state_name[k]:>12}: {n}")

    # --- duplicate-safe unit of splitting -----------------------------
    df["text_key"] = df["review_text"].map(norm)
    # Warn if any duplicate text has inconsistent labels across its rows.
    for key, grp in df.groupby("text_key"):
        if grp["strat"].nunique() > 1:
            print(f"  WARNING: duplicate text with conflicting labels: {key!r}")
    n_dupe_rows = len(df) - df["text_key"].nunique()
    print(f"\nDuplicate review texts: {n_dupe_rows} extra row(s) across "
          f"{df['text_key'].nunique()} unique texts")

    # one representative row per unique text (first occurrence) drives the split
    reps = df.drop_duplicates("text_key", keep="first").copy()

    pinned_keys = set(reps.loc[reps["id"].isin(FEWSHOT_IDS), "text_key"])
    missing = FEWSHOT_IDS - set(df.loc[df["id"].isin(FEWSHOT_IDS), "id"])
    if missing:
        raise ValueError(f"Few-shot ids not found in data: {missing}")

    pool = reps[~reps["text_key"].isin(pinned_keys)]
    train_pool, test_pool = train_test_split(
        pool, test_size=TEST_FRAC, stratify=pool["strat"], random_state=RANDOM_SEED
    )

    train_keys = set(train_pool["text_key"]) | pinned_keys
    test_keys  = set(test_pool["text_key"])
    assert not (train_keys & test_keys), "train/test key overlap"

    df["set"] = df["text_key"].isin(train_keys).astype(int)  # 1=train, 0=test

    # --- report -------------------------------------------------------
    print(f"\nSplit: train(set=1)={int((df['set']==1).sum())}  "
          f"test(set=0)={int((df['set']==0).sum())}  "
          f"(pinned few-shot -> train: {len(pinned_keys)})")
    print("\nPer-state train/test counts:")
    ct = pd.crosstab(df["strat"].map(state_name), df["set"])
    print(ct.to_string())
    print("\nPros/cons (type) by set:")
    print(pd.crosstab(df["type"], df["set"]).to_string())

    out = df.drop(columns=["text_key"])
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(out)} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    sys.exit(main())
