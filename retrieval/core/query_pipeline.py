import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.models.ner.ner_predict import NERPredictor
from retrieval.models.dynamic_weight.dynamic_weight_predict import DynamicWeightPredictor
from retrieval.core.search_engine import SearchEngine
from retrieval.utils.kg_utils import (
    extract_key_risk,
    extract_key_spa,
    kg_plan_relevance_retrieval,
    kg_mileage_relevance_retrieval
)

logger = logging.getLogger(__name__)


class TBMRiskQueryPipeline:
    def __init__(self):
        dynamic_weight_predictor = DynamicWeightPredictor()
        self.search_engine = SearchEngine(dynamic_weight_model=dynamic_weight_predictor)
        self.ner_predictor = NERPredictor()

    def run(self, query: str) -> List[Dict[str, Any]]:
        """
        执行完整检索流程

        Args:
            query: 用户查询文本

        Returns:
            检索结果列表，每个结果包含：
            - node_id: 方案节点ID
            - final_score: 最终得分
            - plan_data: 方案关联信息
            - design_data: 设计/探测信息
        """
        key_spa_mileage, key_spa_line = extract_key_spa(query)
        key_risk = extract_key_risk(query)
        key_geo = self.ner_predictor.predict(query)

        logger.info(f"提取信息 - 里程: {key_spa_mileage}, 线路: {key_spa_line}, 风险: {key_risk}, 地理实体: {key_geo}")

        search_results = self.search_engine.search(input_text=query, key_risk=key_risk, key_geo=key_geo)

        if not search_results:
            logger.warning("未找到匹配的方案")
            return []

        final_results = []
        for result in search_results:
            node_id = result["node_id"]
            final_score = result["final_score"]

            plan_data = kg_plan_relevance_retrieval(node_id)
            design_data = kg_mileage_relevance_retrieval(
                line_name=key_spa_line,
                mileage=key_spa_mileage,
                risk_type=key_risk
            )

            final_results.append({
                "node_id": node_id,
                "final_score": final_score,
                "plan_data": plan_data,
                "design_data": design_data,
                "extracted_info": {
                    "key_spa_mileage": key_spa_mileage,
                    "key_spa_line": key_spa_line,
                    "key_risk": key_risk,
                    "key_geo": key_geo
                }
            })

        logger.info(f"检索完成，返回 {len(final_results)} 条结果")
        return final_results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pipeline = TBMRiskQueryPipeline()
    results = pipeline.run("IV级深埋硬质岩掉块")
    for i, result in enumerate(results, 1):
        print(f"\n=== 结果 {i} ===")
        print(f"方案ID: {result['node_id']}")
        print(f"得分: {result['final_score']:.4f}")
        print(f"提取信息: {result['extracted_info']}")
        print(f"方案数据: {result['plan_data']}")
        print(f"设计数据: {result['design_data']}")
