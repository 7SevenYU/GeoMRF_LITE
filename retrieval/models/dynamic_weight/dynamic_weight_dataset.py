import torch
import sys
from pathlib import Path
from torch.utils.data import Dataset

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.utils import config
from retrieval.models.dynamic_weight.dynamic_weight_data_process import DynamicWeightPreprocess


class DynamicDataset(Dataset):
    def __init__(self, file_path: str):
        processor = DynamicWeightPreprocess(file_path)
        self.data = processor.generate_training_data()

    def __getitem__(self, item):
        score_key = self.data[item]['s_key']
        score_sem = self.data[item]['s_sem']
        loss_keyword = self.data[item]['l_keyword']
        loss_embed = self.data[item]['l_embed']
        return score_key, score_sem, loss_keyword, loss_embed

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    score_key, score_sem, loss_keyword, loss_embed = zip(*batch)
    return (
        torch.tensor(score_key, dtype=torch.float32).to(config.device),
        torch.tensor(score_sem, dtype=torch.float32).to(config.device),
        torch.tensor(loss_keyword, dtype=torch.float32).to(config.device),
        torch.tensor(loss_embed, dtype=torch.float32).to(config.device),
    )
