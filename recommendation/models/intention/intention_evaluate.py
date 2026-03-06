"""
intention_evaluate.py — Evaluate the trained model on the test set.

Computes and reports all metrics cited in the paper:
  Accuracy, Precision, Recall, F1-score, TP, FP, FN, TN.

Usage:
    python -m recommendation.models.intention.intention_evaluate
    python -m recommendation.models.intention.intention_evaluate --test data/intention/test.txt
    python -m recommendation.models.intention.intention_evaluate --model checkpoints/intention_model.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.utils import config
from recommendation.models.intention.intention_dataset import TextDataset, collate_fn
from recommendation.models.intention.intention_model import ClassifyModel


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(
    tp: int, fp: int, fn: int, tn: int
) -> dict[str, float]:
    accuracy  = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def print_report(tp, fp, fn, tn, metrics: dict[str, float]) -> None:
    bar = "─" * 48
    print(f"\n{bar}")
    print("  Intention Recognition — Test-set Evaluation")
    print(bar)
    print(f"  {'Accuracy':<12}: {metrics['accuracy']:.4f}")
    print(f"  {'Precision':<12}: {metrics['precision']:.4f}")
    print(f"  {'Recall':<12}: {metrics['recall']:.4f}")
    print(f"  {'F1-score':<12}: {metrics['f1']:.4f}")
    print(bar)
    # Confusion matrix
    w = 8
    print(f"\n  Confusion Matrix (Predicted →)")
    print(f"  {'':10s}  {'Pos':>{w}}  {'Neg':>{w}}")
    print(f"  {'True Pos':10s}  {tp:>{w}}  {fn:>{w}}   ← actual positive")
    print(f"  {'True Neg':10s}  {fp:>{w}}  {tn:>{w}}   ← actual negative")
    print(f"\n  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{bar}\n")


# ── Evaluation loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: ClassifyModel,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = config.IR_THRESHOLD,
) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN)."""
    model.eval()
    tp = fp = fn = tn = 0

    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)                         # (B,)

        logits = model(x).squeeze(1)                       # (B,)
        preds = (torch.sigmoid(logits) >= threshold).long()

        for pred, true in zip(preds.tolist(), labels.tolist()):
            if   pred == 1 and true == 1: tp += 1
            elif pred == 1 and true == 0: fp += 1
            elif pred == 0 and true == 1: fn += 1
            else:                         tn += 1

    return tp, fp, fn, tn


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the intention-recognition model on the test set."
    )
    parser.add_argument(
        "--test",  default=str(config.IR_TEST_PATH),
        help=f"Test data file (default: {config.IR_TEST_PATH})"
    )
    parser.add_argument(
        "--model", default=str(config.IR_CHECKPOINT_DIR / "intention_model.pt"),
        help="Path to saved model weights"
    )
    parser.add_argument(
        "--threshold", type=float, default=config.IR_THRESHOLD,
        help=f"Sigmoid decision threshold (default: {config.IR_THRESHOLD})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.IR_BATCH_SIZE,
    )
    args = parser.parse_args()

    device = config.device
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Test   : {args.test}")
    print(f"Threshold: {args.threshold}")

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.isfile(args.model):
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            "Run intention_train.py first."
        )
    model = ClassifyModel()
    state = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    # ── Load test data ─────────────────────────────────────────────────────────
    test_set = TextDataset(args.test)
    print(f"Test samples: {len(test_set)}")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    tp, fp, fn, tn = evaluate(model, test_loader, device, threshold=args.threshold)
    metrics = compute_metrics(tp, fp, fn, tn)
    print_report(tp, fp, fn, tn, metrics)


if __name__ == "__main__":
    main()
