"""
intention_dataset.py — Dataset and BERT-embedding utilities.

Changes vs. original:
  - Uses IR_MAX_LENGTH from config (tokenizer truncation was missing).
  - embed_by_bert now accepts a batch of strings for efficiency.
  - collate_fn uses batch encoding (single forward pass per batch).
  - Malformed-line detection splits on the last space to handle
    sentences that themselves contain spaces.
"""

import sys
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Sequence
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.utils import config


class TextDataset(Dataset):
    """Read a corpus file with lines of the form  `<sentence> <0|1>`."""

    def __init__(self, data_path: str):
        self.data: list[tuple[str, int]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 1)          # split on the LAST space
                if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
                    print(f"[Warning] line {lineno} skipped (malformed): {line!r}")
                    continue
                self.data.append((parts[0], int(parts[1])))

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


# ── BERT embedding ─────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_by_bert(text: str | Sequence[str]) -> torch.Tensor:
    """
    Encode one or more sentences with the frozen BERT encoder.

    Args:
        text: a single string  →  returns shape (seq_len, 768)
              a list of strings →  returns shape (batch, max_seq_len, 768),
                                   padded to the longest sequence in the list.
    """
    tokenizer = config.get_tokenizer()
    bert = config.get_bert_model()

    single = isinstance(text, str)
    texts = [text] if single else list(text)

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.IR_MAX_LENGTH,
    ).to(config.device)

    output = bert(**encoded)                         # last_hidden_state: (B, L, 768)
    embeddings = output.last_hidden_state

    return embeddings.squeeze(0) if single else embeddings   # (L,768) or (B,L,768)


# ── Collate ────────────────────────────────────────────────────────────────────

def collate_fn(batch: list[tuple[str, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch collation for DataLoader.

    Encodes all sentences in one BERT forward pass, then pads to the
    longest sequence in the batch.

    Returns:
        padded  — (batch_size, max_seq_len, 768)
        labels  — (batch_size,)  dtype=long
    """
    sentences, labels = zip(*batch)

    # Single batched BERT call — much faster than one call per sentence
    embeddings = embed_by_bert(list(sentences))      # (B, L, 768)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return embeddings, labels_tensor
