"""
GeoMRF 检索引擎 - 统一检索接口

四阶段检索流程：
1. 关键字收缩子图 - 根据风险类型查询相关方案节点
2. 向量过滤 - 计算向量相似度并过滤
3. 动态权重计算 - 融合向量相似度和关键词匹配得分
4. 关联检索 - 查询方案关联数据和设计探测数据
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.utils import config
from retrieval.models.dynamic_weight.dynamic_weight_predict import DynamicWeightPredictor
from retrieval.models.ner.ner_predictor import NERPredictor
from retrieval.utils.kg_utils import (
    extract_key_risk,
    extract_key_spa,
    kg_plan_relevance_retrieval,
    kg_mileage_relevance_retrieval
)

logger = logging.getLogger(__name__)


# ============================================================================
# 工具类
# ============================================================================

class _VectorValidator:
    """向量维度验证器"""

    @staticmethod
    def validate():
        """验证模型向量维度与Neo4j向量维度是否一致"""
        try:
            # 获取模型输出维度
            if config.get_bge_model() is not None:
                model_dim = 1024  # BGE维度
            else:
                model_dim = 768   # BERT维度

            # 检查Neo4j中的向量维度
            graph = config.get_graph()
            query = """
            MATCH (n:紧急响应措施)
            WHERE n.embedding_vector IS NOT NULL
            RETURN n.embedding_vector AS vector
            LIMIT 1
            """
            result = graph.run(query).data()

            if result:
                vector_raw = result[0]['vector']
                if isinstance(vector_raw, list) and len(vector_raw) > 0:
                    neo4j_dim = len(vector_raw[0].split(','))
                    if model_dim != neo4j_dim:
                        logger.error(f"向量维度不匹配: 模型{model_dim}维 vs Neo4j{neo4j_dim}维")
                        raise ValueError(f"向量维度不匹配: 模型{model_dim}维 vs Neo4j{neo4j_dim}维")
                    logger.info(f"向量维度验证通过: {neo4j_dim}维")

        except Exception as e:
            logger.error(f"向量维度验证失败: {e}")
            raise


class _VectorParser:
    """向量解析器"""

    @staticmethod
    def parse(vector_raw) -> Optional[np.ndarray]:
        """
        安全解析Neo4j中的向量字段

        Args:
            vector_raw: Neo4j中的向量字段（可能是字符串列表或numpy数组）

        Returns:
            解析后的numpy数组，失败返回None
        """
        try:
            if isinstance(vector_raw, list) and len(vector_raw) > 0:
                # 格式：["0.005,0.044,..."]
                return np.array(vector_raw[0].split(','), dtype=np.float32)
            elif isinstance(vector_raw, str):
                # 格式："0.005,0.044,..."
                return np.array(vector_raw.split(','), dtype=np.float32)
            else:
                # 已经是numpy数组或其他格式
                return np.array(vector_raw, dtype=np.float32)
        except Exception as e:
            logger.error(f"向量解析失败: {e}")
            return None


class _EntityConverter:
    """实体转换器"""

    @staticmethod
    def entities_to_text(entities: Any) -> str:
        """
        将NER实体列表转换为文本

        Args:
            entities: NER返回的实体列表或文本

        Returns:
            转换后的文本字符串
        """
        if not entities:
            return ""

        if isinstance(entities, list):
            # 实体列表格式：[{"text": "断层", "label": "GEO", ...}, ...]
            texts = []
            for entity in entities:
                if isinstance(entity, dict):
                    text = entity.get("text", "")
                    if text:
                        texts.append(text)
            return " ".join(texts) if texts else ""

        # 已经是文本
        return str(entities).strip()


class _LexiconFallback:
    """词典关键词提取（NER失败时的回退方案）"""

    def __init__(self):
        self._lexicon_extractor = None

    def extract_keywords(self, query: str) -> str:
        """
        使用词典提取关键词（NER失败时的回退）

        Args:
            query: 用户查询文本

        Returns:
            提取的关键词文本，失败返回query本身
        """
        try:
            # 延迟加载词典提取器
            if self._lexicon_extractor is None:
                from kg_construction.core.extraction.lexicon_extractor import LexiconExtractor
                self._lexicon_extractor = LexiconExtractor()

            # 使用词典提取地质术语
            keywords = self._lexicon_extractor.extract(query)

            if keywords:
                logger.info(f"词典提取到关键词: {keywords}")
                return " ".join(keywords)
            else:
                logger.warning("词典未提取到关键词，使用query本身")
                return query

        except Exception as e:
            logger.error(f"词典提取失败: {e}，使用query本身")
            return query


# ============================================================================
# 四阶段处理器
# ============================================================================

class _RiskSubgraphQuerier:
    """阶段1：关键字收缩子图"""

    def __init__(self):
        self.graph = config.get_graph()

    def query(self, key_risk: str) -> Tuple[List, List, List]:
        """
        根据风险类型查询相关方案节点

        Args:
            key_risk: 风险类型

        Returns:
            (node_ids, node_vectors, node_keywords)
        """
        try:
            # 根据风险类型查询
            if key_risk:
                query = """
                MATCH (risk:风险类型 {riskType: $risk_type})<-[:RESPONDS_TO]-(solution:紧急响应措施)
                WHERE solution.embedding_vector IS NOT NULL
                RETURN id(solution) AS nodeId,
                       solution.node_id AS sId,
                       solution.embedding_vector AS embedding_vector,
                       solution.keywords AS keywords
                """
                results = self.graph.run(query, risk_type=key_risk).data()
            else:
                logger.warning("风险类型为空，查询所有方案节点")
                results = self._query_all_nodes()

            if not results:
                logger.info(f"未找到风险类型'{key_risk}'的相关节点")
                return [], [], []

            # 解析结果
            node_ids, node_vectors, node_keywords = [], [], []
            for record in results:
                node_id = record.get('nodeId')
                vector_raw = record.get('embedding_vector')
                keywords_raw = record.get('keywords')

                if not node_id or not vector_raw:
                    continue

                # 解析向量
                vector = _VectorParser.parse(vector_raw)
                if vector is None:
                    logger.warning(f"节点{node_id}的向量解析失败")
                    continue

                # 解析关键词
                if isinstance(keywords_raw, list):
                    keywords = "".join(keywords_raw)
                else:
                    keywords = str(keywords_raw) if keywords_raw else ""

                node_ids.append(node_id)
                node_vectors.append(vector)
                node_keywords.append(keywords)

            logger.info(f"阶段1完成：查询到{len(node_ids)}个节点")
            return node_ids, node_vectors, node_keywords

        except Exception as e:
            logger.error(f"阶段1失败: {e}")
            return [], [], []

    def _query_all_nodes(self) -> List:
        """查询所有方案节点（风险类型为空时的回退）"""
        query = """
        MATCH (solution:紧急响应措施)
        WHERE solution.embedding_vector IS NOT NULL
        RETURN id(solution) AS nodeId,
               solution.node_id AS sId,
               solution.embedding_vector AS embedding_vector,
               solution.keywords AS keywords
        """
        return self.graph.run(query).data()


class _VectorCoarseFilter:
    """阶段2：向量粗过滤"""

    def __init__(self):
        self.bge_model = config.get_bge_model()
        if self.bge_model is None:
            self.tokenizer = config.get_tokenizer()
            self.model = config.get_model()
            self.use_bge = False
        else:
            self.use_bge = True
        self.device = config.device
        self.core_threshold = config.SEARCH_CORE_THRESHOLD

    def filter(self, input_text: str, node_ids: List, node_vectors: List) -> List[Dict]:
        """
        计算向量相似度并过滤

        Args:
            input_text: 输入文本
            node_ids: 节点ID列表
            node_vectors: 节点向量列表

        Returns:
            过滤后的节点列表 [{"node_id": x, "keywords": y, "coarse_similarity": z}, ...]
        """
        try:
            # 计算查询向量
            input_vec = self._get_text_embedding(input_text)
            if np.all(input_vec == 0):
                logger.warning("查询向量为零向量")
                return []

            # 计算相似度
            similarities = cosine_similarity([input_vec], node_vectors)[0]

            # 过滤低相似度节点
            filtered_results = []
            for i, sim in enumerate(similarities):
                if sim > self.core_threshold:
                    filtered_results.append({
                        "node_id": node_ids[i],
                        "coarse_similarity": float(sim),
                        "keywords": ""  # 稍后在阶段3填充
                    })

            # 按相似度排序，取前10个
            filtered_results.sort(key=lambda x: x["coarse_similarity"], reverse=True)
            filtered_results = filtered_results[:10]

            logger.info(f"阶段2完成：从{len(node_ids)}个节点过滤到{len(filtered_results)}个")
            return filtered_results

        except Exception as e:
            logger.error(f"阶段2失败: {e}")
            return []

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """获取文本向量"""
        try:
            if self.use_bge:
                embedding = self.bge_model.encode([text], normalize_embeddings=True)[0]
                return embedding
            else:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                       padding=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                return embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"向量计算失败: {e}")
            if self.use_bge:
                return np.zeros(1024)
            else:
                return np.zeros(768)


class _DynamicWeightCalculator:
    """阶段3：动态权重计算"""

    def __init__(self):
        self.dynamic_weight_model = DynamicWeightPredictor()
        self.vector_filter = _VectorCoarseFilter()
        self.lexicon_fallback = _LexiconFallback()

    def calculate(self, query: str, nodes: List[Dict], extracted_info: Dict) -> List[Dict]:
        """
        计算动态权重并融合得分

        Args:
            query: 用户查询
            nodes: 阶段2过滤后的节点列表
            extracted_info: 提取的信息（包含key_geo等）

        Returns:
            融合得分后的节点列表
        """
        try:
            # 处理key_geo（三层容错层级1）
            key_geo_text = _EntityConverter.entities_to_text(extracted_info.get('key_geo', []))
            if not key_geo_text:
                logger.warning("NER未提取到实体，使用词典提取作为fallback")
                key_geo_text = self.lexicon_fallback.extract_keywords(query)

            # 获取alpha和beta（三层容错层级2）
            node_ids = [n["node_id"] for n in nodes]
            try:
                alpha, beta = self.dynamic_weight_model.predict(query, node_ids)
                logger.info(f"动态权重预测: alpha={alpha:.4f}, beta={beta:.4f}")
            except Exception as e:
                logger.error(f"动态权重预测失败: {e}，使用固定权重")
                alpha, beta = config.SEARCH_ALPHA, config.SEARCH_BETA

            # 计算key_geo向量
            key_geo_vec = self.vector_filter._get_text_embedding(key_geo_text)
            if np.all(key_geo_vec == 0):
                logger.warning("key_geo向量为零向量，所有key_score将为0")

            # 为每个节点计算最终得分（三层容错层级3）
            results = []
            for node in nodes:
                # 计算key_score
                try:
                    node_key_vec = self.vector_filter._get_text_embedding(node["keywords"])
                    if np.all(node_key_vec == 0):
                        key_score = 0.0
                    else:
                        key_score = cosine_similarity([key_geo_vec], [node_key_vec])[0][0]
                except Exception as e:
                    logger.warning(f"节点{node['node_id']}的key_score计算失败: {e}")
                    key_score = 0.0

                # 融合得分
                final_score = alpha * node["coarse_similarity"] + beta * key_score

                results.append({
                    "node_id": node["node_id"],
                    "final_score": float(final_score),
                    "score_breakdown": {
                        "coarse_similarity": float(node["coarse_similarity"]),
                        "key_score": float(key_score),
                        "alpha": float(alpha),
                        "beta": float(beta)
                    }
                })

            # 按最终得分排序
            results.sort(key=lambda x: x["final_score"], reverse=True)
            logger.info(f"阶段3完成：计算了{len(results)}个节点的最终得分")
            return results

        except Exception as e:
            logger.error(f"阶段3失败: {e}")
            return []


class _AssociationRetriever:
    """阶段4：关联检索"""

    def __init__(self):
        self.graph = config.get_graph()

    def retrieve(self, nodes: List[Dict], extracted_info: Dict) -> List[Dict]:
        """
        查询关联数据并增强结果

        Args:
            nodes: 阶段3得分后的节点列表
            extracted_info: 提取的信息

        Returns:
            增强后的完整结果
        """
        try:
            results = []
            for node in nodes:
                node_id = node["node_id"]

                # 查询方案数据
                plan_data = self._query_plan_data(node_id)

                # 查询设计数据
                design_data = self._query_design_data(
                    extracted_info.get('key_spa_line'),
                    extracted_info.get('key_spa_mileage'),
                    extracted_info.get('key_risk')
                )

                results.append({
                    "node_id": node_id,
                    "final_score": node["final_score"],
                    "score_breakdown": node.get("score_breakdown", {}),
                    "plan_data": plan_data,
                    "design_data": design_data,
                    "extracted_info": extracted_info
                })

            logger.info(f"阶段4完成：增强了{len(results)}个节点的关联数据")
            return results

        except Exception as e:
            logger.error(f"阶段4失败: {e}")
            # 即使关联数据查询失败，也返回基本结果
            return [{
                "node_id": n["node_id"],
                "final_score": n["final_score"],
                "score_breakdown": n.get("score_breakdown", {}),
                "plan_data": {},
                "design_data": {},
                "extracted_info": extracted_info
            } for n in nodes]

    def _query_plan_data(self, node_id: int) -> Dict:
        """查询方案关联数据"""
        try:
            return kg_plan_relevance_retrieval(node_id)
        except Exception as e:
            logger.warning(f"节点{node_id}的方案数据查询失败: {e}")
            return {}

    def _query_design_data(self, line_name: str, mileage: str, risk_type: str) -> Dict:
        """查询设计探测数据"""
        try:
            return kg_mileage_relevance_retrieval(
                line_name=line_name,
                mileage=mileage,
                risk_type=risk_type
            )
        except Exception as e:
            logger.warning(f"设计数据查询失败: {e}")
            return {}


# ============================================================================
# 主编排器
# ============================================================================

class SearchEngine:
    """统一检索引擎"""

    def __init__(self, top_k: int = None):
        """
        初始化检索引擎

        Args:
            top_k: 返回结果数量，默认使用config.SEARCH_TOP_K
        """
        logger.info("=" * 60)
        logger.info("初始化SearchEngine")
        logger.info("=" * 60)

        # 验证向量维度
        try:
            _VectorValidator.validate()
        except Exception as e:
            logger.warning(f"向量维度验证失败，继续使用: {e}")

        # 初始化四个阶段
        self.stage1 = _RiskSubgraphQuerier()
        self.stage2 = _VectorCoarseFilter()
        self.stage3 = _DynamicWeightCalculator()
        self.stage4 = _AssociationRetriever()

        # 初始化NER预测器
        self.ner_predictor = NERPredictor()

        # 配置参数
        self.top_k = top_k if top_k is not None else config.SEARCH_TOP_K

        logger.info(f"SearchEngine初始化完成，top_k={self.top_k}")
        logger.info("=" * 60)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        统一的检索入口

        Args:
            query: 用户查询文本

        Returns:
            检索结果列表，每个结果包含：
            - node_id: 方案节点ID
            - final_score: 最终得分
            - score_breakdown: 得分分解
            - plan_data: 方案关联数据
            - design_data: 设计探测数据
            - extracted_info: 提取的信息
        """
        if not query or not query.strip():
            logger.warning("查询文本为空")
            return []

        logger.info(f"开始检索: {query}")

        # 1. 信息提取
        extracted_info = self._extract_info(query)
        logger.info(f"提取信息: 风险={extracted_info['key_risk']}, "
                   f"里程={extracted_info['key_spa_mileage']}, "
                   f"线路={extracted_info['key_spa_line']}, "
                   f"实体数={len(extracted_info['key_geo'])}")

        # 2. 四阶段检索
        # 阶段1：关键字收缩子图
        node_ids, node_vectors, node_keywords = self.stage1.query(extracted_info['key_risk'])
        if not node_ids:
            logger.info("阶段1未找到节点，结束检索")
            return []

        # 为节点添加关键词（供阶段3使用）
        for i, node in enumerate(node_ids):
            if i < len(node_keywords):
                # 这里需要在阶段2的结果中添加keywords
                pass

        # 阶段2：向量过滤
        # 需要将node_vectors传递给阶段2
        filtered_nodes = self.stage2.filter(query, node_ids, node_vectors)
        if not filtered_nodes:
            logger.info("阶段2过滤后无节点，结束检索")
            return []

        # 为过滤后的节点添加keywords
        for i, node in enumerate(filtered_nodes):
            if i < len(node_keywords):
                node["keywords"] = node_keywords[i]

        # 阶段3：动态权重计算
        scored_nodes = self.stage3.calculate(query, filtered_nodes, extracted_info)
        if not scored_nodes:
            logger.info("阶段3计算失败，结束检索")
            return []

        # 阶段4：关联检索
        results = self.stage4.retrieve(scored_nodes, extracted_info)

        # 取top_k结果
        results = results[:self.top_k]

        logger.info(f"检索完成，返回{len(results)}条结果")
        return results

    def _extract_info(self, query: str) -> Dict[str, Any]:
        """
        提取查询中的关键信息

        Returns:
            包含以下字段的字典：
            - key_risk: 风险类型
            - key_spa_mileage: 里程
            - key_spa_line: 线路
            - key_geo: 地理实体列表
        """
        # 提取里程和线路
        key_spa_mileage, key_spa_line = extract_key_spa(query)

        # 提取风险类型
        key_risk = extract_key_risk(query)

        # 提取地理实体（NER）
        try:
            key_geo = self.ner_predictor.predict(query)
        except Exception as e:
            logger.error(f"NER预测失败: {e}")
            key_geo = []

        return {
            'key_risk': key_risk or "",
            'key_spa_mileage': key_spa_mileage,
            'key_spa_line': key_spa_line,
            'key_geo': key_geo
        }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    engine = SearchEngine()
    results = engine.search("IV级深埋硬质岩掉块")

    for i, result in enumerate(results, 1):
        print(f"\n{'=' * 60}")
        print(f"结果 {i}")
        print(f"{'=' * 60}")
        print(f"方案ID: {result['node_id']}")
        print(f"最终得分: {result['final_score']:.4f}")
        print(f"得分分解: {result.get('score_breakdown', {})}")
        print(f"提取信息: {result['extracted_info']}")
        print(f"方案数据: {result['plan_data']}")
        print(f"设计数据: {result['design_data']}")
