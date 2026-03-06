"""
intention_model.py — BiLSTM + MLP classifier.

Architecture (unchanged from paper):
  BiLSTM (hidden=256 per direction) → temporal max-pool → MLP(512→128→1)
  Output: raw logit (apply sigmoid externally for probability).
"""

import sys
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.utils import config


class ClassifyModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = config.IR_HIDDEN_DIM,
        mlp_dim: int = config.IR_MLP_DIM,
        dropout: float = config.IR_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.pool = nn.AdaptiveMaxPool1d(1)     # temporal max-pooling
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),              # raw logit
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, 768)
        Returns:
            logits: (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)              # (B, L, hidden*2)
        pooled = self.pool(
            lstm_out.transpose(1, 2)            # (B, hidden*2, L)
        ).squeeze(-1)                           # (B, hidden*2)
        return self.head(pooled)                # (B, 1)
