"""
Shared scoring for the two-dimension (schedule / job control) classifier.

Single source of truth so the fine-tuned model (12.evaluate_multilabel.py) and the
GPT benchmark (13.GPT_multilabel_benchmark.R -> scored via 12 --preds_csv) are scored
identically.

Label convention: each review carries two independent binary flags,
    schedule  in {0,1}   (dimension 1, the primary/headline dimension)
    jobctrl   in {0,1}   (dimension 2, secondary/exploratory)
The model emits two digits "<schedule><jobctrl>" (e.g. "10").
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    classification_report,
    confusion_matrix,
)

STATE_NAMES = ["none(00)", "jobctrl(01)", "sched(10)", "both(11)"]  # index = 2*s + c


def parse_two_digit(text):
    """Parse a model output into (schedule, jobctrl, well_formed).

    Permissive: pulls digit characters out of whatever the model produced, maps any
    non-zero digit to 1, and takes the first two. `well_formed` is False when fewer
    than two digits were found (missing positions default to 0) so callers can log
    malformed generations, mirroring the warning in the binary 07.evaluate_qwen.py.
    """
    digits = [1 if (c.isdigit() and int(c) != 0) else 0 for c in str(text) if c.isdigit()]
    sched = digits[0] if len(digits) >= 1 else 0
    ctrl = digits[1] if len(digits) >= 2 else 0
    return sched, ctrl, len(digits) >= 2


def _bootstrap_f1_ci(y_true, y_pred, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for binary F1 (positive class = 1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = f1_score(y_true[idx], y_pred[idx], pos_label=1, zero_division=0)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def _dim_row(name, y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    lo, hi = _bootstrap_f1_ci(y_true, y_pred)
    n_pos = int(np.sum(np.asarray(y_true) == 1))
    return {
        "dimension": name, "precision": p, "recall": r, "f1": f,
        "f1_ci95": (lo, hi), "n_pos": n_pos,
    }


def multilabel_report(true_sched, true_ctrl, pred_sched, pred_ctrl,
                      name="model", target_f1=0.95, n_malformed=0):
    """Print and return the full multi-label metric bundle."""
    ts = np.asarray(true_sched, dtype=int)
    tc = np.asarray(true_ctrl, dtype=int)
    ps = np.asarray(pred_sched, dtype=int)
    pc = np.asarray(pred_ctrl, dtype=int)

    sched = _dim_row("schedule", ts, ps)
    ctrl = _dim_row("job_control", tc, pc)

    macro_f1 = 0.5 * (sched["f1"] + ctrl["f1"])
    micro_f1 = f1_score(np.concatenate([ts, tc]), np.concatenate([ps, pc]),
                        pos_label=1, average="binary", zero_division=0)
    exact_match = float(np.mean((ps == ts) & (pc == tc)))

    true_state = 2 * ts + tc
    pred_state = 2 * ps + pc

    print("\n" + "=" * 66)
    print(f"MULTI-LABEL REPORT — {name}   (n={len(ts)}, malformed outputs={n_malformed})")
    print("=" * 66)
    print(f"{'dimension':<14}{'precision':>11}{'recall':>9}{'F1':>9}{'F1 95% CI':>18}{'n_pos':>7}")
    for row in (sched, ctrl):
        lo, hi = row["f1_ci95"]
        print(f"{row['dimension']:<14}{row['precision']:>11.4f}{row['recall']:>9.4f}"
              f"{row['f1']:>9.4f}{f'[{lo:.3f},{hi:.3f}]':>18}{row['n_pos']:>7}")
    print("-" * 66)
    print(f"macro-F1 (2 dims): {macro_f1:.4f}   micro-F1: {micro_f1:.4f}   "
          f"exact-match: {exact_match:.4f}")

    status = "PASS" if sched["f1"] >= target_f1 else "below target"
    print(f"\nHEADLINE — schedule F1 = {sched['f1']:.4f}  (target {target_f1:.2f}: {status})")

    print("\n4-way state classification report:")
    labels = [0, 1, 2, 3]
    print(classification_report(
        true_state, pred_state, labels=labels, target_names=STATE_NAMES, zero_division=0
    ))
    print("Per-dimension confusion (rows=true, cols=pred, order [0,1]):")
    print(f"  schedule:\n{confusion_matrix(ts, ps, labels=[0, 1])}")
    print(f"  job_control:\n{confusion_matrix(tc, pc, labels=[0, 1])}")

    return {
        "name": name,
        "schedule": {k: sched[k] for k in ("precision", "recall", "f1", "f1_ci95", "n_pos")},
        "job_control": {k: ctrl[k] for k in ("precision", "recall", "f1", "f1_ci95", "n_pos")},
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "exact_match": exact_match,
        "n": int(len(ts)),
        "n_malformed": int(n_malformed),
    }
