import torch
import torch.nn as nn
import sys
from pathlib import Path
from torchcrf import CRF
import math

from transformers import BertModel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.utils import config


class LatticeSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None, lexicon_links=None):
        B, L, H = hidden_states.size()
        Q = self.q(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, self.num_heads, scores.size(2), -1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        if lexicon_links is not None:
            for b in range(B):
                for ch_idx, word_idxs in lexicon_links[b].items():
                    for word_idx in word_idxs:
                        scores[b, :, int(ch_idx), word_idx] += 1.5

        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, V)
        context = context.transpose(1, 2).contiguous().view(B, L, H)
        return self.o(context)


class LatticeBERT(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # 使用共享的BERT模型
        bert_path = project_root / "models" / "bert-base-chinese"
        self.bert = self.bert = BertModel.from_pretrained(str(bert_path))
        self.attn = LatticeSelfAttention(self.bert.config.hidden_size, self.bert.config.num_attention_heads)
        self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, labels=None, label_mask=None,
                lexicon_links=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True
        )
        x = outputs.last_hidden_state
        x = self.attn(x, attention_mask=attention_mask, lexicon_links=lexicon_links)
        x = self.dropout(self.norm(x))
        logits = self.classifier(x)

        if labels is not None:
            loss = -self.crf(logits[:, 1:], labels[:, 1:], mask=label_mask[:, 1:].bool(), reduction='mean')
            return {'loss': loss, 'logits': logits[:, 1:, :]}
        else:
            prediction = self.crf.decode(logits[:, 1:], mask=label_mask[:, 1:].bool())
            return {'logits': prediction}
