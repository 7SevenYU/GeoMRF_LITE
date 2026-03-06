import os
import sys
import logging
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.models.ner.ner_model import LatticeBERT
from retrieval.models.ner.ner_data_process import LatticeBERTPreprocessor

from retrieval.utils import config

logger = logging.getLogger(__name__)


class NERPredictor:
    def __init__(self, model_path=None):
        self.device = config.device
        if model_path is None:
            model_path = config.NER_CHECKPOINT_DIR / "ner_model.pt"
        # 初始化预处理器（不使用文件路径）
        self.preprocessor = LatticeBERTPreprocessor(file_path=None)

        # 加载模型
        self.model = LatticeBERT(num_labels=len(self.preprocessor.label_list))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 标签映射
        self.id2label = {i: label for label, i in self.preprocessor.label_map.items()}
        self.id2label[0] = 'O'

    def _prepare_sentence(self, sentence):
        dummy_labels = ['O'] * len(sentence)
        processed = self.preprocessor.process_sample(sentence, dummy_labels)

        input_dict = {
            'input_ids': torch.tensor([processed['input_ids']], dtype=torch.long).to(self.device),
            'attention_mask': torch.tensor([processed['attention_mask']], dtype=torch.long).to(self.device),
            'token_type_ids': torch.tensor([processed['token_type_ids']], dtype=torch.long).to(self.device),
            'position_ids': torch.tensor([processed['position_ids']], dtype=torch.long).to(self.device),
            'label_mask': torch.tensor([processed['label_mask']], dtype=torch.bool).to(self.device),
            'lexicon_links': [processed['lexicon_links']]
        }
        return input_dict, processed["text"]

    def _extract_entities(self, chars, labels, target_prefix='GEO'):
        entities = []
        current_entity = ''
        for ch, label in zip(chars, labels):
            if label == f'B-{target_prefix}':
                if current_entity:
                    entities.append(current_entity)
                current_entity = ch
            elif label == f'I-{target_prefix}':
                if current_entity:
                    current_entity += ch
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = ''
        if current_entity:
            entities.append(current_entity)
        return entities

    def predict(self, sentence):
        input_dict, raw_text = self._prepare_sentence(sentence)

        with torch.no_grad():
            output = self.model(
                input_ids=input_dict['input_ids'],
                attention_mask=input_dict['attention_mask'],
                token_type_ids=input_dict['token_type_ids'],
                position_ids=input_dict['position_ids'],
                label_mask=input_dict['label_mask'],
                lexicon_links=input_dict['lexicon_links']
            )
            pred_ids = output['logits'][0]
            char_len = len(raw_text)
            pred_labels = [self.id2label[i] for i in pred_ids[1:1 + char_len]]
        extracted = "".join(self._extract_entities(raw_text, pred_labels))
        if not extracted:
            extracted = " "
            logger.warning(f"No entities extracted for {raw_text}")
        return extracted


def main():
    recognizer = NERPredictor()
    while True:
        text = input("输入文本（输入 exit 退出）：").strip()
        if text.lower() == 'exit':
            break
        print(recognizer.predict(text))


if __name__ == '__main__':
    main()
