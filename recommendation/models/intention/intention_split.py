"""
intention_split.py — Split the raw corpus into train / val / test.

Split ratio: 6 : 2 : 2 (stratified by label to preserve class balance).
A fixed random seed (IR_SEED = 42) guarantees reproducibility.

Auto-balance:
  Automatically samples IR_POS_TARGET positives and IR_NEG_TARGET negatives
  from the deduplicated corpus before splitting, so the caller never needs
  to manually balance the source file.
  - If a class has fewer samples than the target, ALL available samples are
    used and a warning is printed.
  - Targets are defined in config (IR_POS_TARGET / IR_NEG_TARGET).

Usage:
    python -m recommendation.models.intention.intention_split
    python -m recommendation.models.intention.intention_split --src data/intention/intention_recognition.txt
    python -m recommendation.models.intention.intention_split --pos 1000 --neg 1000
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.utils import config


def load_corpus(path: str) -> list[tuple[str, int]]:
    """
    Read 'sentence<space>label' lines; skip blank / malformed lines.
    Deduplicates by sentence text before returning (leakage prevention).
    """
    samples: list[tuple[str, int]] = []
    seen: dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
                print(f"[Warning] line {lineno} skipped (malformed): {line!r}")
                continue
            text, label = parts[0], int(parts[1])
            if text in seen:
                if seen[text] != label:
                    print(f"[Warning] line {lineno} duplicate with conflicting label, skipped: {text!r}")
                continue
            seen[text] = label
            samples.append((text, label))

    return samples


def balance_corpus(
    samples: list[tuple[str, int]],
    pos_target: int,
    neg_target: int,
    seed: int,
) -> list[tuple[str, int]]:
    """
    Randomly draw exactly pos_target positives and neg_target negatives.
    If a class has fewer samples than requested, use all available and warn.
    """
    rng = random.Random(seed)

    positives = [s for s in samples if s[1] == 1]
    negatives = [s for s in samples if s[1] == 0]

    if len(positives) < pos_target:
        print(f"[Warning] Only {len(positives)} positive samples available "
              f"(target={pos_target}). Using all.")
        pos_target = len(positives)

    if len(negatives) < neg_target:
        print(f"[Warning] Only {len(negatives)} negative samples available "
              f"(target={neg_target}). Using all.")
        neg_target = len(negatives)

    selected_pos = rng.sample(positives, pos_target)
    selected_neg = rng.sample(negatives, neg_target)

    balanced = selected_pos + selected_neg
    rng.shuffle(balanced)
    return balanced


def stratified_split(
    samples: list[tuple[str, int]],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Stratified split preserving label distribution in every subset.
    Uses round() to avoid silent sample loss on odd-sized groups.
    """
    rng = random.Random(seed)

    by_label: dict[int, list] = defaultdict(list)
    for item in samples:
        by_label[item[1]].append(item)

    train, val, test = [], [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * train_ratio)
        n_val   = round(n * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    for split in (train, val, test):
        rng.shuffle(split)

    return train, val, test


def verify_no_leakage(train, val, test) -> None:
    train_texts = {s for s, _ in train}
    val_texts   = {s for s, _ in val}
    test_texts  = {s for s, _ in test}
    tv = train_texts & val_texts
    tt = train_texts & test_texts
    vt = val_texts   & test_texts
    if tv or tt or vt:
        print(f"[ERROR] Data leakage: Train/Val={len(tv)}, "
              f"Train/Test={len(tt)}, Val/Test={len(vt)}")
    else:
        print("  [OK] No overlap between splits (data leakage: 0)")


def save_split(samples: list[tuple[str, int]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sentence, label in samples:
            f.write(f"{sentence} {label}\n")
    print(f"  Saved {len(samples):>6} samples → {path}")


def report(train, val, test) -> None:
    def label_counts(split):
        pos = sum(1 for _, l in split if l == 1)
        return pos, len(split) - pos

    print("\n── Split statistics ──────────────────────────────────")
    for name, split in (("Train", train), ("Val", val), ("Test", test)):
        pos, neg = label_counts(split)
        print(f"  {name:5s}: {len(split):4d} total | pos={pos} neg={neg}")
    print(f"  {'Total':5s}: {len(train)+len(val)+len(test):4d}")
    print("──────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(description="Split intention-recognition corpus.")
    parser.add_argument("--src",         default=str(config.IR_DATA_DIR / "intention_recognition.txt"))
    parser.add_argument("--train",       default=str(config.IR_TRAIN_PATH))
    parser.add_argument("--val",         default=str(config.IR_VAL_PATH))
    parser.add_argument("--test",        default=str(config.IR_TEST_PATH))
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio",   type=float, default=0.2)
    parser.add_argument("--seed",        type=int,   default=config.IR_SEED)
    parser.add_argument("--pos",         type=int,   default=1500,
                        help="Number of positive samples to use (default: 1500)")
    parser.add_argument("--neg",         type=int,   default=1500,
                        help="Number of negative samples to use (default: 1500)")
    args = parser.parse_args()

    # ── Load & deduplicate ────────────────────────────────────────────────────
    print(f"Loading corpus from: {args.src}")
    all_samples = load_corpus(args.src)
    pos_all = sum(1 for _, l in all_samples if l == 1)
    neg_all = sum(1 for _, l in all_samples if l == 0)
    print(f"After dedup: {len(all_samples)} total | pos={pos_all} neg={neg_all}")

    # ── Auto-balance ──────────────────────────────────────────────────────────
    print(f"\nSampling: pos={args.pos}, neg={args.neg} (seed={args.seed})")
    samples = balance_corpus(all_samples, args.pos, args.neg, args.seed)
    pos_sel = sum(1 for _, l in samples if l == 1)
    neg_sel = sum(1 for _, l in samples if l == 0)
    print(f"Selected:  {len(samples)} total | pos={pos_sel} neg={neg_sel}")

    # ── Split ─────────────────────────────────────────────────────────────────
    train, val, test = stratified_split(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    report(train, val, test)
    verify_no_leakage(train, val, test)

    save_split(train, args.train)
    save_split(val,   args.val)
    save_split(test,  args.test)
    print("Done.")


if __name__ == "__main__":
    main()
