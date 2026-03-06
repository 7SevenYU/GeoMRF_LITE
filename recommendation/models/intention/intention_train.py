"""
intention_recognition_train.py — Training entry point.

Key improvements vs. original:
  - Adds a validation loop; best model is saved based on val accuracy
    (not train accuracy), which is the standard for held-out evaluation.
  - Reads train / val paths from config; run split script first.
  - All hyper-parameters come from config (single source of truth).
  - set_seed also fixes torch.backends.cudnn for full reproducibility.
  - Training summary printed at the end.

Usage:
    python intention_recognition_split.py   # create splits first
    python intention_recognition_train.py
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.utils import config
from .intention_dataset import TextDataset, collate_fn
from .intention_model import ClassifyModel


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = config.IR_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── One epoch helpers ──────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    optimizer: optim.Optimizer | None = None,
    threshold: float = config.IR_THRESHOLD,
) -> tuple[float, float]:
    """
    Run one train or eval epoch.

    If `optimizer` is provided the model is set to train mode and
    weights are updated; otherwise eval mode (no grad).

    Returns:
        (avg_loss, accuracy)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    phase = "Train" if is_train else "Val"

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        bar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [{phase}]", leave=False)
        for x, labels in bar:
            x = x.to(device)
            labels = labels.to(device).float().unsqueeze(1)   # (B, 1)

            logits = model(x)                                  # (B, 1)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.IR_GRAD_CLIP
                )
                optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= threshold)
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed()
    os.makedirs(config.IR_SAVE_PATH, exist_ok=True)
    device = config.device
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_set = TextDataset(config.IR_TRAIN_PATH)
    val_set = TextDataset(config.IR_VAL_PATH)
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    loader_kwargs = dict(
        collate_fn=collate_fn,
        batch_size=config.IR_BATCH_SIZE,
        num_workers=0,           # BERT singleton is not fork-safe; keep 0
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ClassifyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.IR_LR)
    criterion = nn.BCEWithLogitsLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    save_path = os.path.join(config.IR_SAVE_PATH, "intention_model.pt")

    print(f"\nStarting training for {config.IR_EPOCHS} epochs …\n")
    for epoch in range(1, config.IR_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, epoch, optimizer=optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, epoch
        )
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model  (val acc = {best_val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
