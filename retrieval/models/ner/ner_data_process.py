import json
import sys
from collections import defaultdict
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.utils import config


class LatticeBERTPreprocessor:
    def __init__(self, file_path, max_seq_len=128):
        self.tokenizer = config.get_tokenizer()
        self.max_seq_len = max_seq_len
        self.file_path = file_path
        # 从 txt 文件加载 lexicon 列表
        lexicon_file = project_root / "data" / "lexicon.txt"
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            self.lexicon = [line.strip() for line in f if line.strip()]
        self.label_list = ['O', 'B-GEO', 'I-GEO']
        self.label_map = {label: i for i, label in enumerate(self.label_list)}

    def match_lexicon(self, sentence):
        matches = []
        n = len(sentence)
        for start in range(n):
            for end in range(start + 1, n + 1):
                word = sentence[start:end]
                if word in self.lexicon:
                    matches.append((start, end, word))
        return matches

    def process_sample(self, text, labels):
        char_tokens = list(text)
        matched_words = self.match_lexicon(text)
        lex_tokens = [w for _, _, w in matched_words]
        # 限制最大长度
        if len(char_tokens) + len(lex_tokens) >= self.max_seq_len - 2:
            char_tokens = char_tokens[:self.max_seq_len - 2 - len(lex_tokens)]

        # 重新匹配词图
        matched_words = [(s, e, w) for (s, e, w) in self.match_lexicon(text) if s < len(char_tokens)]
        lex_tokens = [w for _, _, w in matched_words]
        lex_start = [s + 1 for s, _, _ in matched_words]  # 注意 +1（对齐 CLS）

        # 添加开始和结束符
        input_tokens = ["[CLS]"] + char_tokens + lex_tokens + ["[SEP]"]
        # 构建序列
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        token_type_ids = [0] + [0] * len(char_tokens) + [1] * len(lex_tokens) + [0]
        position_ids = list(range(len(char_tokens) + 1)) + lex_start + [len(char_tokens) + len(lex_tokens) + 1]
        labels = [0] + [self.label_map.get(i, 0) for i in labels[:len(char_tokens)]] + [0] * len(lex_tokens) + [0]
        # 起始符[CLS]的label_mask必须为1，以初始化路径分数
        label_mask = [1] + [1] * len(char_tokens) + [0] * len(lex_tokens) + [0]
        attention_mask = [1] + [1] * len(char_tokens) + [1] * len(lex_tokens) + [1]

        lexicon_links = defaultdict(list)
        for i, (s, _, _) in enumerate(matched_words):
            lexicon_links[s + 1].append(len(char_tokens) + i + 1)

        # 句子填充为最大长度
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            token_type_ids += [0] * pad_len
            position_ids += [0] * pad_len
            labels += [0] * pad_len
            label_mask += [0] * pad_len
            attention_mask += [0] * pad_len

        return {
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "labels": labels,
            "label_mask": label_mask,
            "lexicon_links": dict(lexicon_links),
        }

    def process_batch(self):
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    text = sample.get("text", "")
                    labels = sample.get("labels", [])
                    processed = self.process_sample(text, labels)
                    data.append(processed)
        return data


if __name__ == "__main__":
    process = LatticeBERTPreprocessor(file_path=config.NER_DATA)
    data = process.process_batch()
    print(f"成功加载 {len(data)} 条训练数据")
    for i in data[:3]:
        print(f"文本: {i['text']}")
        print(f"标签: {i['labels'][:len(i['text'])]}")
        print("-" * 50)
