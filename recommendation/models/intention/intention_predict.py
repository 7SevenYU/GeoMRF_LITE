"""
intention_recognition_predict.py — Runtime inference.

IntentionPredictor.predict(text) → float  (probability of positive class)
IntentionPredictor.classify(text) → bool  (True = mitigation-relevant)
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.utils import config
from .intention_dataset import embed_by_bert
from .intention_model import ClassifyModel


class IntentionPredictor:
    """Thin wrapper that loads the trained model and exposes predict / classify."""

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = config.IR_THRESHOLD,
    ):
        self.device = config.device
        self.threshold = threshold
        self.model = ClassifyModel()

        if model_path is None:
            model_path = os.path.join(config.IR_SAVE_PATH, "intention_model.pt")
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model weights not found: {model_path}\n"
                "Run intention_recognition_train.py first."
            )
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> float:
        """Return the probability of the positive (mitigation-relevant) class."""
        emb = embed_by_bert(text).unsqueeze(0).to(self.device)  # (1, L, 768)
        logits = self.model(emb)                                  # (1, 1)
        return torch.sigmoid(logits).item()

    def classify(self, text: str) -> bool:
        """Return True if the query is mitigation-relevant (prob ≥ threshold)."""
        return self.predict(text) >= self.threshold


def main() -> None:
    recognizer = IntentionPredictor()
    print(f"Model loaded. Decision threshold: {recognizer.threshold}")
    print("Type 'exit' to quit.\n")

    while True:
        text = input("Input text: ").strip()
        if text.lower() == "exit":
            break
        prob = recognizer.predict(text)
        label = "POSITIVE (mitigation-relevant)" if prob >= recognizer.threshold else "NEGATIVE"
        print(f"  Probability : {prob:.4f}")
        print(f"  Label       : {label}\n")


if __name__ == "__main__":
    main()
