import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.models.ner.ner_data_process import LatticeBERTPreprocessor


class LatticeNERDataset(Dataset):
    def __init__(self, file_path):
        self.processor = LatticeBERTPreprocessor(file_path)
        self.data = self.processor.process_batch()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(sample["token_type_ids"], dtype=torch.long),
            "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
            "label_mask": torch.tensor(sample["label_mask"], dtype=torch.long),
            "lexicon_links": sample["lexicon_links"],  # dict 类型，batch 时处理
            "text": sample["text"]
        }


def collate_fn(batch: List[Dict]) -> Dict:
    batch_size = len(batch)
    keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels", "label_mask"]
    batch_tensors = {key: torch.stack([item[key] for item in batch]) for key in keys}
    lexicon_links = [item["lexicon_links"] for item in batch]
    texts = [item["text"] for item in batch]
    return {
        **batch_tensors,
        "lexicon_links": lexicon_links,
        "text": texts
    }
