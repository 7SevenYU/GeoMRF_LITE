"""
recommendation/utils/config.py — 推荐模块配置
"""

import os
import sys
import torch
from pathlib import Path
from transformers import BertTokenizer, BertModel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入共享的BERT配置
from retrieval.utils import config as shared_config

# ── 意图识别配置 ───────────────────────────────────────────────────────────────────
IR_BERT_MODEL: str = "bert-base-chinese"
IR_MAX_LENGTH: int = 128
IR_HIDDEN_DIM: int = 256
IR_MLP_DIM: int = 128
IR_DROPOUT: float = 0.3
IR_BATCH_SIZE: int = 16
IR_LR: float = 3e-4
IR_EPOCHS: int = 5
IR_GRAD_CLIP: float = 1.0
IR_SEED: int = 42
IR_THRESHOLD: float = 0.6

# ── 路径配置 ───────────────────────────────────────────────────────────────────────
IR_CHECKPOINT_DIR = project_root / "recommendation" / "models" / "intention" / "checkpoints"
IR_DATA_DIR = project_root / "recommendation" / "data" / "intention"
IR_TRAIN_PATH = IR_DATA_DIR / "train.txt"
IR_VAL_PATH = IR_DATA_DIR / "val.txt"
IR_TEST_PATH = IR_DATA_DIR / "test.txt"

# 反馈记录文件
FEEDBACK_FILE = IR_DATA_DIR / "feedback_records.json"
MAX_REJECTIONS = 3

# ── 设备配置 ───────────────────────────────────────────────────────────────────────
device = shared_config.device

# ── 共享的BERT模型（使用retrieval模块的配置）────────────────────────────────────────
_tokenizer: BertTokenizer | None = None
_bert_model: BertModel | None = None


def get_tokenizer() -> BertTokenizer:
    """获取BERT tokenizer（使用共享的）"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = shared_config.get_tokenizer()
    return _tokenizer


def get_bert_model() -> BertModel:
    """获取BERT模型（使用共享的，frozen）"""
    global _bert_model
    if _bert_model is None:
        _bert_model = shared_config.get_model()
    return _bert_model


# 向后兼容
def get_model() -> BertModel:
    return get_bert_model()
