import os.path
import sys
from pathlib import Path

import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent  # dynamic_weight -> models -> retrieval -> project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.models.dynamic_weight.dynamic_weight_data_process import DynamicWeightPreprocess
from retrieval.models.dynamic_weight.dynamic_weight_model import DynamicWeighting
from retrieval.utils import config


class SubclassWithIDPreprocess(DynamicWeightPreprocess):
    def __init__(self):
        super().__init__("")

    def _load_entities(self, risk_type=None, id_list=None):
        """按给定 node_id 列表筛选紧急响应措施节点"""
        if not id_list:
            return [], []
        query = """
        MATCH (n:紧急响应措施)
        WHERE n.node_id IN $id_list
          AND n.keywords IS NOT NULL
          AND n.embedding_vector IS NOT NULL
        RETURN n.keywords AS keywords, n.embedding_vector AS vector
        """
        data = self.graph.run(query, id_list=id_list).data()
        keyword_list = [record["keywords"] for record in data]
        vector_list = [np.array(record["vector"][0].split(','), dtype=np.float32) for record in data]
        return keyword_list, vector_list


class DynamicWeightPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(config.DYNAMIC_SAVE_PATH, "dynamic_weight_model.pt")
        self.preprocessor = SubclassWithIDPreprocess()
        self.model = DynamicWeighting()
        self.model.load_state_dict(torch.load(model_path, map_location=config.device))
        self.model.eval()
        self.model.to(config.device)

    def predict(self, q: str, id_list=None):
        # 获取关键词匹配度和语义模糊度
        score_key, score_sem = self.preprocessor.score_query(query=q, id_list=id_list)
        # 转换为张量并送入模型
        skey = torch.tensor([score_key], dtype=torch.float32).to(config.device)
        ssem = torch.tensor([score_sem], dtype=torch.float32).to(config.device)
        with torch.no_grad():
            alpha, beta = self.model(skey, ssem)
        alpha = alpha.item()
        beta = beta.item()
        return alpha, beta


# 示例使用
if __name__ == '__main__':
    predictor = DynamicWeightPredictor()
    query = "TBM发生岩爆"
    result = predictor.predict(query, [1, 2, 3])
    print(result)
