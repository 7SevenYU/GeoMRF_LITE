import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.utils import config
from retrieval.models.dynamic_weight.dynamic_weight_predict import DynamicWeightPredictor

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, dynamic_weight_model, core_threshold=0.6, top_k=4, alpha=0.5, beta=0.5):
        self.graph = config.get_graph()
        self.bge_model = config.get_bge_model()
        if self.bge_model is None:
            self.tokenizer = config.get_tokenizer()
            self.model = config.get_model()
            self.use_bge = False
        else:
            self.use_bge = True

        self.core_threshold = core_threshold
        self.top_k = top_k
        self.dynamic_weight_model = dynamic_weight_model
        self.alpha = alpha
        self.beta = beta
        self.device = config.device
        logging.basicConfig(level=logging.INFO)

    def _get_text_embedding(self, text: str) -> np.ndarray:
        if not text:
            raise ValueError("输入文本不能为空")
        try:
            if self.use_bge:
                embedding = self.bge_model.encode([text], normalize_embeddings=True)[0]
                return embedding
            else:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                return embedding.cpu().numpy()
        except Exception as e:
            logging.error(f"生成文本向量失败: {e}")
            if self.use_bge:
                return np.zeros(1024)
            else:
                return np.zeros(self.model.config.hidden_size)

    def _get_kg_embedding(self, key_risk: str):
        query = """
        MATCH (risk:风险类型 {riskType: $risk_type})<-[:RESPONDS_TO]-(solution:紧急响应措施)
        WHERE solution.embedding_vector IS NOT NULL
        RETURN solution.node_id AS nodeId, solution.embedding_vector AS embedding_vector, solution.keywords AS keywords
        """
        try:
            results = self.graph.run(query, risk_type=key_risk).data()
        except Exception as e:
            logging.warning(f"查询图谱失败: {e}, 尝试获取所有方案节点")
            query = """
            MATCH (solution:紧急响应措施)
            WHERE solution.embedding_vector IS NOT NULL
            RETURN solution.node_id AS nodeId, solution.embedding_vector AS embedding_vector, solution.keywords AS keywords
            """
            try:
                results = self.graph.run(query).data()
            except Exception as e:
                logging.error(f"图数据库连接失败: {e}")
                return [], [], []

        if not results:
            return [], [], []

        node_ids, node_vectors, node_keywords = [], [], []
        for record in results:
            node_id = record.get('nodeId')
            vector_raw = record.get('embedding_vector')
            keywords_raw = record.get('keywords')

            if not node_id or not vector_raw:
                continue
            try:
                vector = json.loads(vector_raw) if isinstance(vector_raw, str) else vector_raw
                keywords = "".join(keywords_raw) if isinstance(keywords_raw, list) else str(keywords_raw)
                node_ids.append(node_id)
                node_vectors.append(np.array(vector, dtype=np.float64))
                node_keywords.append(keywords)
            except Exception as e:
                logging.warning(f"解析节点 {node_id} 的信息失败: {e}")

        return node_ids, node_vectors, node_keywords

    def _find_most_similar_nodes(self, input_text: str, key_risk: str):
        node_ids, node_vectors, node_keywords = self._get_kg_embedding(key_risk)

        if not node_vectors:
            return [], [], []

        input_vec = self._get_text_embedding(input_text)
        if np.all(input_vec == 0):
            return [], [], []

        try:
            similarities = cosine_similarity([input_vec], node_vectors)[0]
        except Exception as e:
            logging.error(f"计算相似度失败: {e}")
            return [], [], []

        top_indices = [i for i, sim in enumerate(similarities) if sim > self.core_threshold]
        top_indices = sorted(top_indices, key=lambda i: similarities[i], reverse=True)[:10]

        selected_ids = [node_ids[i] for i in top_indices]
        selected_keywords = [node_keywords[i] for i in top_indices]
        selected_sims = [similarities[i] for i in top_indices]

        return selected_sims, selected_ids, selected_keywords

    def search(self, input_text: str, key_risk: str, key_geo: str):
        if not input_text or not key_risk or not key_geo:
            raise ValueError("input_text、key_risk、key_geo 参数均不能为空")

        coarse_sims, node_ids, node_keywords = self._find_most_similar_nodes(input_text, key_risk)
        if not node_ids:
            logging.info("没有找到匹配的方案节点")
            return []

        alpha, beta = self.dynamic_weight_model.predict(input_text, node_ids)
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta

        key_geo_vec = self._get_text_embedding(key_geo)
        if np.all(key_geo_vec == 0):
            logging.warning("关键地理信息向量为零向量，key_score 将为 0")

        results = []
        for sim, nid, keywords in zip(coarse_sims, node_ids, node_keywords):
            node_key_vec = self._get_text_embedding(keywords)
            if np.all(node_key_vec == 0):
                key_score = 0
            else:
                try:
                    key_score = cosine_similarity([key_geo_vec], [node_key_vec])[0][0]
                except Exception as e:
                    logging.warning(f"计算 key_score 失败: {e}")
                    key_score = 0

            results.append({
                "node_id": nid,
                "coarse_similarity": float(sim),
                "key_score": float(key_score)
            })

        final_scores = [
            {
                "node_id": r["node_id"],
                "final_score": float(alpha) * r["coarse_similarity"] + float(beta) * r["key_score"]
            }
            for r in results
        ]
        return sorted(final_scores, key=lambda x: x["final_score"], reverse=True)[:self.top_k]


if __name__ == '__main__':
    predictor = DynamicWeightPredictor()
    search = SearchEngine(dynamic_weight_model=predictor)
    try:
        result = search.search("IV级深埋硬质岩掉块", "掉块", ".")
        print(result)
    except Exception as e:
        logging.error(f"搜索过程中出现异常: {e}")
