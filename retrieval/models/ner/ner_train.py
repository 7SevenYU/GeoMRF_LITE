import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from transformers import BertConfig, BertTokenizer

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.models.ner.ner_lexicon import build_lattice_lexicon_items, build_lattice_links, load_lexicon_from_dataset
from retrieval.utils import config


DEFAULT_LABELS = ["O", "B-GEO", "I-GEO"]


def resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0].lower() == base_dir.name.lower():
        return (base_dir.parent / p).resolve()
    return (base_dir / p).resolve()


def load_label_list(path: Path) -> List[str]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return data
    return DEFAULT_LABELS


class JsonlNERDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        tokenizer: BertTokenizer,
        label2id: Dict[str, int],
        lexicon_trie,
        max_seq_len: int = 128,
        max_lexicon_nodes: int = 64,
    ) -> None:
        self.samples: List[Dict[str, torch.Tensor]] = []
        self.bad_rows = 0

        with data_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    self.bad_rows += 1
                    continue

                text = item.get("text", "")
                labels = item.get("labels", [])
                if not isinstance(text, str) or not isinstance(labels, list):
                    self.bad_rows += 1
                    continue

                if not text or not labels:
                    self.bad_rows += 1
                    continue

                # Robust alignment for unexpected dirty rows.
                max_chars = min(len(text), len(labels), max_seq_len - 2)
                if max_chars <= 0:
                    self.bad_rows += 1
                    continue

                chars = list(text[:max_chars])
                token_labels = labels[:max_chars]

                input_tokens = ["[CLS]"] + chars + ["[SEP]"]
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

                token_type_ids = [0] * len(input_ids)
                attention_mask = [1] * len(input_ids)
                position_ids = list(range(len(input_ids)))

                label_ids = [label2id["O"]]
                for tag in token_labels:
                    label_ids.append(label2id.get(tag, label2id["O"]))
                label_ids.append(label2id["O"])

                # 1 only on real character positions.
                label_mask = [0] + [1] * len(token_labels) + [0]

                self.samples.append(
                    {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "position_ids": torch.tensor(position_ids, dtype=torch.long),
                        "labels": torch.tensor(label_ids, dtype=torch.long),
                        "label_mask": torch.tensor(label_mask, dtype=torch.long),
                        "lexicon_items": build_lattice_lexicon_items(
                            text=text,
                            trie=lexicon_trie,
                            tokenizer=tokenizer,
                            max_chars=max_chars,
                            max_lexicon_nodes=max_lexicon_nodes,
                        ),
                        "line_no": line_no,
                    }
                )

        if not self.samples:
            raise ValueError(f"No valid samples loaded from: {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels", "label_mask"]
    out = {
        k: torch.nn.utils.rnn.pad_sequence(
            [x[k] for x in batch],
            batch_first=True,
            padding_value=0,
        )
        for k in keys
    }
    char_seq_len = out["input_ids"].size(1)
    out["lexicon_items"] = [x["lexicon_items"] for x in batch]
    out["lexicon_links"] = [build_lattice_links(char_seq_len, x["lexicon_items"]) for x in batch]
    return out


class LatticeSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None, lexicon_links=None):
        bsz, seq_len, hidden = hidden_states.size()
        q = self.q(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.expand(-1, self.num_heads, scores.size(2), -1)
            scores = scores.masked_fill(attn == 0, -1e9)

        if lexicon_links is not None:
            for b in range(bsz):
                for ch_idx, word_idxs in lexicon_links[b].items():
                    for word_idx in word_idxs:
                        if 0 <= int(ch_idx) < seq_len and 0 <= int(word_idx) < seq_len:
                            scores[b, :, int(ch_idx), int(word_idx)] += 1.5

        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.o(context)


class LatticeBERT(nn.Module):
    def __init__(self, config: BertConfig, num_labels: int, pretrained_model_dir: Path = None) -> None:
        super().__init__()
        from transformers import BertModel

        if pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(str(pretrained_model_dir), config=config)
        else:
            self.bert = BertModel(config)
        self.attn = LatticeSelfAttention(config.hidden_size, config.num_attention_heads)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def _build_lexicon_hidden(self, lexicon_items, char_seq_len: int, device):
        hidden_size = self.bert.config.hidden_size
        max_nodes = max((len(items) for items in lexicon_items), default=0)
        if max_nodes == 0:
            batch_size = len(lexicon_items)
            return (
                torch.zeros(batch_size, 0, hidden_size, device=device),
                torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            )

        batch_hidden = []
        batch_mask = []
        for items in lexicon_items:
            node_hidden = []
            for item in items:
                token_ids = torch.tensor(item["token_ids"], dtype=torch.long, device=device)
                word_embed = self.bert.embeddings.word_embeddings(token_ids).mean(dim=0)
                position_id = min(int(item["position_id"]), char_seq_len - 1)
                pos_tensor = torch.tensor([position_id], dtype=torch.long, device=device)
                pos_embed = self.bert.embeddings.position_embeddings(pos_tensor).squeeze(0)
                type_tensor = torch.tensor([1], dtype=torch.long, device=device)
                type_embed = self.bert.embeddings.token_type_embeddings(type_tensor).squeeze(0)
                lex_hidden = self.bert.embeddings.LayerNorm(word_embed + pos_embed + type_embed)
                lex_hidden = self.bert.embeddings.dropout(lex_hidden)
                node_hidden.append(lex_hidden)

            if node_hidden:
                sample_hidden = torch.stack(node_hidden, dim=0)
            else:
                sample_hidden = torch.zeros(0, hidden_size, device=device)

            pad_nodes = max_nodes - sample_hidden.size(0)
            if pad_nodes > 0:
                sample_hidden = torch.cat(
                    [sample_hidden, torch.zeros(pad_nodes, hidden_size, device=device)],
                    dim=0,
                )
            batch_hidden.append(sample_hidden)
            batch_mask.append([1] * len(items) + [0] * (max_nodes - len(items)))

        return (
            torch.stack(batch_hidden, dim=0),
            torch.tensor(batch_mask, dtype=torch.long, device=device),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, labels=None, label_mask=None, lexicon_links=None, lexicon_items=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True,
        )
        x = outputs.last_hidden_state
        char_seq_len = input_ids.size(1)

        combined_attention_mask = attention_mask
        combined_label_mask = label_mask
        combined_labels = labels
        if lexicon_items is not None:
            lexicon_hidden, lexicon_mask = self._build_lexicon_hidden(
                lexicon_items=lexicon_items,
                char_seq_len=char_seq_len,
                device=input_ids.device,
            )
            if lexicon_hidden.size(1) > 0:
                x = torch.cat([x, lexicon_hidden], dim=1)
                combined_attention_mask = torch.cat([attention_mask, lexicon_mask], dim=1)
                if label_mask is not None:
                    zeros = torch.zeros(label_mask.size(0), lexicon_hidden.size(1), dtype=label_mask.dtype, device=label_mask.device)
                    combined_label_mask = torch.cat([label_mask, zeros], dim=1)
                if labels is not None:
                    zeros = torch.zeros(labels.size(0), lexicon_hidden.size(1), dtype=labels.dtype, device=labels.device)
                    combined_labels = torch.cat([labels, zeros], dim=1)
                if lexicon_links is None:
                    lexicon_links = [build_lattice_links(char_seq_len, items) for items in lexicon_items]

        x = self.attn(x, attention_mask=combined_attention_mask, lexicon_links=lexicon_links)
        x = self.dropout(self.norm(x))
        logits = self.classifier(x)

        if combined_label_mask is None:
            combined_label_mask = (combined_attention_mask > 0) & (torch.cat([token_type_ids, torch.ones_like(combined_attention_mask[:, token_type_ids.size(1):])], dim=1) == 0)
            combined_label_mask[:, 0] = 0
            last_idx = attention_mask.sum(dim=1) - 1
            for b in range(combined_label_mask.size(0)):
                if last_idx[b] >= 0:
                    combined_label_mask[b, last_idx[b]] = 0

        if combined_labels is not None:
            loss = -self.crf(logits[:, 1:, :], combined_labels[:, 1:], mask=combined_label_mask[:, 1:].bool(), reduction="mean")
            return {"loss": loss}

        prediction = self.crf.decode(logits[:, 1:, :], mask=combined_label_mask[:, 1:].bool())
        return {"pred_ids": prediction}


def train(
    data_path: Path,
    model_dir: Path,
    labels_path: Path,
    save_path: Path,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 3e-5,
    max_seq_len: int = 128,
    max_lexicon_nodes: int = 64,
) -> None:
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    label_list = load_label_list(labels_path)
    label2id = {label: i for i, label in enumerate(label_list)}
    if "O" not in label2id:
        raise ValueError(f"labels.json must include 'O', got: {label_list}")

    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    lexicon_trie = load_lexicon_from_dataset(data_path)
    dataset = JsonlNERDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        label2id=label2id,
        lexicon_trie=lexicon_trie,
        max_seq_len=max_seq_len,
        max_lexicon_nodes=max_lexicon_nodes,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    config = BertConfig.from_pretrained(str(model_dir))
    model = LatticeBERT(config, num_labels=len(label_list), pretrained_model_dir=model_dir)

    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    print(f"loaded samples: {len(dataset)}, skipped bad rows: {dataset.bad_rows}")
    print(f"label schema: {label_list}")
    print(f"lexicon terms: {lexicon_trie.term_count}")

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = (batch["attention_mask"] > 0).long().to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device).bool()
            lexicon_links = batch["lexicon_links"]

            max_label_id = int(labels.max().item())
            if max_label_id >= len(label_list):
                raise ValueError(
                    f"Label id out of range: max label id={max_label_id}, num_labels={len(label_list)}. "
                    f"Please check labels.json and dataset labels."
                )

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                labels=labels,
                label_mask=label_mask,
                lexicon_links=lexicon_links,
                lexicon_items=batch["lexicon_items"],
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step == 1 or step % 50 == 0:
                print(f"epoch={epoch + 1} step={step} loss={loss.item():.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "label_list": label_list,
        "max_seq_len": max_seq_len,
        "max_lexicon_nodes": max_lexicon_nodes,
        "uses_lexicon_links": True,
    }
    torch.save(checkpoint, str(save_path))
    print(f"saved model: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Lattice-BERT-CRF on processed jsonl data.")
    parser.add_argument("--data-path", default=None, help="Path to training data (default: use config.NER_DATA)")
    parser.add_argument("--labels", default=None, help="Path to labels.json (default: ner_data/labels.json)")
    parser.add_argument("--model-dir", default=None, help="Path to BERT model (default: use project models/bert-base-chinese)")
    parser.add_argument("--model-path", default=None, help="Path to save model (default: config.NER_SAVE_PATH/lattice_bert_crf.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default: config.NER_EPOCHS)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: config.NER_BATCH_SIZE)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: config.NER_LR)")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-lexicon-nodes", type=int, default=64)
    args = parser.parse_args()

    data_path = Path(args.data_path) if args.data_path else config.NER_DATA
    labels_path = Path(args.labels) if args.labels else Path(__file__).parent / "ner_data" / "labels.json"
    model_dir = Path(args.model_dir) if args.model_dir else config.get_project_root() / "models" / "bert-base-chinese"
    model_path = Path(args.model_path) if args.model_path else config.NER_SAVE_PATH / "lattice_bert_crf.pt"

    epochs = args.epochs if args.epochs is not None else config.NER_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else config.NER_BATCH_SIZE
    lr = args.lr if args.lr is not None else config.NER_LR

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found: {labels_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"BERT model directory not found: {model_dir}")

    print("=" * 60)
    print("Lattice-BERT-CRF NER Training")
    print("=" * 60)
    print(f"[Config] Data path: {data_path}")
    print(f"[Config] Labels: {labels_path}")
    print(f"[Config] BERT model: {model_dir}")
    print(f"[Config] Save path: {model_path}")
    print(f"[Config] Epochs: {epochs}")
    print(f"[Config] Batch size: {batch_size}")
    print(f"[Config] Learning rate: {lr}")
    print("=" * 60)

    train(
        data_path=data_path,
        labels_path=labels_path,
        model_dir=model_dir,
        save_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_seq_len=args.max_seq_len,
        max_lexicon_nodes=args.max_lexicon_nodes,
    )


if __name__ == "__main__":
    main()
