"""
recommendation/core/recommendation_engine.py — 推荐引擎核心逻辑
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.models.intention import IntentionPredictor
from retrieval.core.query_pipeline import TBMRiskQueryPipeline

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """TBM地质风险防控推荐引擎"""

    def __init__(self):
        """初始化推荐引擎"""
        self.intention_predictor = IntentionPredictor()
        self.query_pipeline = TBMRiskQueryPipeline()
        self.state = {
            "results": [],
            "index": 0,
            "first_query": "",
            "rejection_count": 0,
            "active": False
        }
        logger.info("推荐引擎初始化完成")

    def should_recommend(self, query: str) -> bool:
        """
        判断查询是否需要推荐（意图识别）

        Args:
            query: 用户查询文本

        Returns:
            True表示需要推荐，False表示不需要
        """
        return self.intention_predictor.classify(query)

    def recommend(self, query: str) -> List[Dict[str, Any]]:
        """
        获取推荐结果列表

        Args:
            query: 用户查询文本

        Returns:
            推荐结果列表，每个结果包含：
            - node_id: 方案节点ID
            - final_score: 最终得分
            - plan_data: 方案关联信息
            - design_data: 设计/探测信息
            - extracted_info: 提取的信息
        """
        return self.query_pipeline.run(query)

    def get_current_recommendation(self) -> Optional[Dict[str, Any]]:
        """
        获取当前推荐方案

        Returns:
            当前推荐结果的字典，如果没有则返回None
        """
        if self.state["active"] and self.state["index"] < len(self.state["results"]):
            return self.state["results"][self.state["index"]]
        return None

    def next_recommendation(self) -> Optional[Dict[str, Any]]:
        """
        切换到下一个备选方案（用户拒绝时）

        Returns:
            下一个推荐结果的字典，如果没有更多则返回None
        """
        if self.state["index"] < len(self.state["results"]) - 1:
            self.state["index"] += 1
            current = self.get_current_recommendation()
            logger.info(f"切换到方案 {self.state['index'] + 1}/{len(self.state['results'])}")
            return current
        logger.warning("已无更多备选方案")
        return None

    def start_recommendation(self, query: str) -> bool:
        """
        开始新的推荐流程

        Args:
            query: 用户查询文本

        Returns:
            True表示成功开始推荐，False表示没有找到推荐结果
        """
        results = self.recommend(query)
        if not results:
            logger.warning(f"未找到推荐结果: {query}")
            return False

        self.state["results"] = results
        self.state["index"] = 0
        self.state["first_query"] = query
        self.state["rejection_count"] = 0
        self.state["active"] = True

        logger.info(f"开始推荐流程，找到 {len(results)} 条结果")
        return True

    def increment_rejection_count(self) -> int:
        """
        增加拒绝计数

        Returns:
            当前拒绝次数
        """
        self.state["rejection_count"] += 1
        return self.state["rejection_count"]

    def reset(self):
        """重置推荐状态"""
        self.state = {
            "results": [],
            "index": 0,
            "first_query": "",
            "rejection_count": 0,
            "active": False
        }
        logger.info("推荐状态已重置")

    def get_state(self) -> Dict[str, Any]:
        """获取当前推荐状态（只读）"""
        return {
            "active": self.state["active"],
            "first_query": self.state["first_query"],
            "total_results": len(self.state["results"]),
            "current_index": self.state["index"],
            "rejection_count": self.state["rejection_count"],
            "has_next": self.state["index"] < len(self.state["results"]) - 1
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    engine = RecommendationEngine()

    # 测试
    test_query = "IV级深埋硬质岩掉块"
    print(f"测试查询: {test_query}")
    print(f"是否需要推荐: {engine.should_recommend(test_query)}")

    if engine.start_recommendation(test_query):
        current = engine.get_current_recommendation()
        print(f"当前推荐: {current['node_id']}, 得分: {current['final_score']:.4f}")

        # 测试下一个方案
        next_rec = engine.next_recommendation()
        if next_rec:
            print(f"下一个推荐: {next_rec['node_id']}, 得分: {next_rec['final_score']:.4f}")
