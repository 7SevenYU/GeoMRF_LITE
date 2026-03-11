import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent  # dynamic_weight -> models -> retrieval -> project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.utils import config
from retrieval.models.dynamic_weight.dynamic_weight_dataset import DynamicDataset, collate_fn
from retrieval.models.dynamic_weight.dynamic_weight_model import DynamicWeighting


def compute_entropy(alpha, beta):
    # Add epsilon to avoid log(0)
    eps = 1e-8
    entropy = -alpha * torch.log(alpha + eps) - beta * torch.log(beta + eps)
    return entropy


def train(dataloader, model, optimizer, lambda_entropy, epoch):
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        s_key, s_sem, l_keyword, l_embed = batch

        alpha, beta = model(s_key, s_sem)

        # 主损失：融合得分误差
        fusion_loss = 1 - (alpha * s_key + beta * s_sem)

        # 熵正则项（鼓励解释性）
        entropy_reg = compute_entropy(alpha, beta)

        loss = fusion_loss + lambda_entropy * entropy_reg

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        total_loss += loss.mean().item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
    return avg_loss


def main():
    print("=" * 60)
    print("Dynamic Weight Training")
    print("=" * 60)

    print(f"\n[Config] Batch Size: {config.DW_BATCH_SIZE}")
    print(f"[Config] Epochs: {config.DW_EPOCHS}")
    print(f"[Config] Learning Rate: {config.DW_LR}")
    print(f"[Config] Lambda Entropy: {config.LAMBDA_ENTROPY}")
    print(f"[Config] Device: {config.device}")

    train_dataset = DynamicDataset(file_path=config.DYNAMIC_DATA_FILE)
    print(f"\n[Data] Training samples: {len(train_dataset)}")

    dataloader = DataLoader(dataset=train_dataset, batch_size=config.DW_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = DynamicWeighting().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.DW_LR)

    lambda_entropy = config.LAMBDA_ENTROPY  # 可调范围：0.01 ~ 0.1

    model.train()
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    for epoch in range(1, config.DW_EPOCHS + 1):
        train(dataloader, model, optimizer, lambda_entropy, epoch)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # 保存模型
    os.makedirs(config.DYNAMIC_SAVE_PATH, exist_ok=True)
    save_path = os.path.join(config.DYNAMIC_SAVE_PATH, "dynamic_weight_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
