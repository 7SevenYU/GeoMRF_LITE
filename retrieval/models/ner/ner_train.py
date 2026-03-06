import os
import sys
import torch
from pathlib import Path
from transformers import AdamW
from torch.utils.data import DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.models.ner.ner_dataset import LatticeNERDataset, collate_fn
from retrieval.models.ner.ner_model import LatticeBERT
from retrieval.utils import config


# === 训练流程 ===
def ner_train():
    dataset = LatticeNERDataset(file_path=config.NER_DATA)
    dataloader = DataLoader(dataset, batch_size=config.NER_BATCH_SIZE, collate_fn=collate_fn)
    model = LatticeBERT(num_labels=3)
    optimizer = AdamW(model.parameters(), lr=config.NER_LR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(1, config.NER_EPOCHS + 1):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = (batch['attention_mask'] > 0).long().to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            position_ids = batch['position_ids'].to(device)
            labels = batch['labels'].to(device)
            label_mask = batch['label_mask'].to(device).bool()
            lexicon_links = batch['lexicon_links']
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                labels=labels,
                label_mask=label_mask,
                lexicon_links=lexicon_links
            )
            loss = out['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Loss: {loss.item():.4f}")
    os.makedirs(config.NER_CHECKPOINT_DIR, exist_ok=True)
    save_path = config.NER_CHECKPOINT_DIR / 'ner_model.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    ner_train()
