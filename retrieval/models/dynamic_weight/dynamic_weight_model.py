import torch
import torch.nn as nn


class DynamicWeighting(nn.Module):
    def __init__(self, hidden_dim=8):
        super(DynamicWeighting, self).__init__()
        self.linear1 = nn.Linear(2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, skey, ssem):
        x = torch.stack([skey, ssem], dim=1)  # shape: (B, 2)
        x = self.linear1(x)
        x = torch.relu(x)
        logits = self.linear2(x)  # shape: (B, 2)
        weights = self.softmax(logits)  # shape: (B, 2)
        alpha = weights[:, 0]
        beta = weights[:, 1]
        return alpha, beta
