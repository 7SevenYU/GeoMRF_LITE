"""
recommendation/core/feedback_handler.py — 用户反馈处理
"""

import json
import re
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.utils import config

logger = logging.getLogger(__name__)


class FeedbackHandler:
    """用户反馈处理器"""

    def __init__(self, feedback_file: Optional[Path] = None):
        """
        初始化反馈处理器

        Args:
            feedback_file: 反馈记录文件路径，默认使用配置文件中的路径
        """
        self.feedback_file = feedback_file or config.FEEDBACK_FILE
        self.rejection_keywords = [
            "拒绝", "不行", "不合适", "换一个", "不满意", "没用", "不接受",
            "否定", "不建议", "不推荐", "不考虑", "别的方案", "别的建议",
        ]
        self.rejection_patterns = [
            r"不(行|合适|满意|推荐|接受|准确|合理|考虑)",
            r"没(用|帮助|效果)",
            r"换(一个|个方案)?",
            r"有没有(别的|其他|更好|不同)"
        ]

    def is_rejection(self, text: str) -> bool:
        """
        检测文本是否为拒绝反馈

        Args:
            text: 用户输入文本

        Returns:
            True表示是拒绝反馈，False表示不是
        """
        # 关键词匹配
        if any(keyword in text for keyword in self.rejection_keywords):
            return True

        # 正则表达式匹配
        if any(re.search(pattern, text) for pattern in self.rejection_patterns):
            return True

        return False

    def save_feedback(
        self,
        query: str,
        results: List[Dict[str, Any]],
        rejection_count: int = 0,
        comment: str = ""
    ) -> bool:
        """
        保存用户反馈到JSON文件

        Args:
            query: 用户查询
            results: 推荐结果列表
            rejection_count: 拒绝次数
            comment: 附加评论

        Returns:
            True表示保存成功，False表示失败
        """
        feedback_entry = {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "node_id": r.get("node_id"),
                    "final_score": r.get("final_score")
                }
                for r in results[:5]  # 只保存前5个结果的摘要
            ],
            "rejection_count": rejection_count,
            "comment": comment,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            # 读取现有数据
            if self.feedback_file.exists():
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"反馈文件损坏，将创建新文件: {self.feedback_file}")
                        data = []
            else:
                data = []

            # 确保父目录存在
            self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

            # 添加新记录
            data.append(feedback_entry)

            # 写入文件
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"反馈已保存: {query}")
            return True

        except Exception as e:
            logger.error(f"保存反馈失败: {e}")
            return False

    def load_feedback(self) -> List[Dict[str, Any]]:
        """
        加载所有反馈记录

        Returns:
            反馈记录列表，如果文件不存在或读取失败则返回空列表
        """
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"加载反馈失败: {e}")
            return []

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        获取反馈统计信息

        Returns:
            包含统计信息的字典
        """
        feedback_list = self.load_feedback()

        if not feedback_list:
            return {
                "total_count": 0,
                "avg_rejection_count": 0,
                "avg_results_count": 0
            }

        total_count = len(feedback_list)
        total_rejections = sum(f.get("rejection_count", 0) for f in feedback_list)
        total_results = sum(f.get("results_count", 0) for f in feedback_list)

        return {
            "total_count": total_count,
            "avg_rejection_count": total_rejections / total_count if total_count > 0 else 0,
            "avg_results_count": total_results / total_count if total_count > 0 else 0
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试反馈处理器
    handler = FeedbackHandler()

    # 测试拒绝检测
    test_cases = [
        "这个方案不行",
        "换一个吧",
        "挺好的",
        "有没有更好的建议"
    ]

    for text in test_cases:
        is_rej = handler.is_rejection(text)
        print(f"'{text}' -> {'拒绝' if is_rej else '接受'}")

    # 测试保存反馈
    handler.save_feedback(
        query="测试查询",
        results=[
            {"node_id": "test1", "final_score": 0.9},
            {"node_id": "test2", "final_score": 0.8}
        ],
        rejection_count=2
    )

    # 测试统计
    stats = handler.get_feedback_stats()
    print(f"反馈统计: {stats}")
