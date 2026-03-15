"""
FeedbackAnalyzer - 反馈分析器

功能：
1. 分析用户拒绝的原因
2. 提供调整建议
3. 判断是否可以切换到备选方案
"""

import logging
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class RejectionReason(Enum):
    """拒绝原因分类"""
    UNSUITABLE_CONDITION = "unsuitable_condition"      # 适用条件不符
    INSUFFICIENT_MEASURES = "insufficient_measures"    # 措施不足
    TOO_COMPLEX = "too_complex"                        # 方案过于复杂
    TOO_SIMPLE = "too_simple"                          # 方案过于简单
    COST_CONCERN = "cost_concern"                      # 成本顾虑
    TIME_CONCERN = "time_concern"                      # 时间顾虑
    SAFETY_CONCERN = "safety_concern"                  # 安全顾虑
    ALREADY_TRIED = "already_tried"                    # 已尝试过
    NEED_MORE_INFO = "need_more_info"                  # 需要更多信息
    OTHER = "other"                                    # 其他原因


class FeedbackAnalyzer:
    """
    反馈分析器

    分析用户反馈，提供调整建议
    """

    # 关键词映射到拒绝原因
    REJECTION_KEYWORDS = {
        RejectionReason.UNSUITABLE_CONDITION: [
            "条件不符", "不适用", "情况不同", "地质条件", "不适合"
        ],
        RejectionReason.INSUFFICIENT_MEASURES: [
            "措施不够", "措施不足", "不够全面", "需要更多", "缺少"
        ],
        RejectionReason.TOO_COMPLEX: [
            "太复杂", "难以实施", "操作困难", "太麻烦", "不实用"
        ],
        RejectionReason.TOO_SIMPLE: [
            "太简单", "不够详细", "需要更具体", "不够深入"
        ],
        RejectionReason.COST_CONCERN: [
            "成本", "费用", "预算", "太贵", "经济"
        ],
        RejectionReason.TIME_CONCERN: [
            "时间", "太慢", "工期", "进度", "效率"
        ],
        RejectionReason.SAFETY_CONCERN: [
            "不安全", "危险", "风险", "安全隐患", "安全"
        ],
        RejectionReason.ALREADY_TRIED: [
            "试过", "用过", "做过", "已经", "无效"
        ],
        RejectionReason.NEED_MORE_INFO: [
            "不清楚", "不明白", "详细", "解释", "说明"
        ]
    }

    def __init__(self):
        """初始化反馈分析器"""
        logger.info("反馈分析器初始化")

    def analyze_rejection(self, user_feedback: str) -> Dict[str, Any]:
        """
        分析用户拒绝原因

        Args:
            user_feedback: 用户反馈内容

        Returns:
            分析结果字典，包含原因类型和建议
        """
        if not user_feedback:
            return {
                "reason": RejectionReason.OTHER,
                "confidence": 0.0,
                "suggestions": ["请提供更多反馈信息"]
            }

        # 检测关键词
        detected_reasons = []
        for reason, keywords in self.REJECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in user_feedback:
                    detected_reasons.append((reason, keyword))
                    break

        if not detected_reasons:
            # 未检测到明确原因
            return {
                "reason": RejectionReason.OTHER,
                "confidence": 0.0,
                "suggestions": [
                    "请详细说明不推荐该方案的具体原因",
                    "可以描述当前实际情况与方案不符的地方"
                ]
            }

        # 取第一个检测到的原因
        primary_reason, keyword = detected_reasons[0]

        # 生成建议
        suggestions = self._generate_suggestions(primary_reason)

        logger.info(f"拒绝原因分析: reason={primary_reason.value}, keyword={keyword}")
        return {
            "reason": primary_reason,
            "confidence": 0.8 if len(detected_reasons) == 1 else 0.6,
            "keyword": keyword,
            "suggestions": suggestions
        }

    def _generate_suggestions(self, reason: RejectionReason) -> List[str]:
        """
        根据拒绝原因生成建议

        Args:
            reason: 拒绝原因

        Returns:
            建议列表
        """
        suggestions_map = {
            RejectionReason.UNSUITABLE_CONDITION: [
                "建议查看更符合当前地质条件的方案",
                "可以提供更详细的现场情况以便推荐更合适的方案"
            ],
            RejectionReason.INSUFFICIENT_MEASURES: [
                "可以寻找包含更多防控措施的方案",
                "建议组合多个方案的综合措施"
            ],
            RejectionReason.TOO_COMPLEX: [
                "建议寻找更简化、易实施的方案",
                "可以分阶段实施复杂方案"
            ],
            RejectionReason.TOO_SIMPLE: [
                "建议寻找更详细、更具体的方案",
                "可以寻求专家指导或更全面的方案"
            ],
            RejectionReason.COST_CONCERN: [
                "建议寻找性价比更高的方案",
                "可以分优先级实施关键措施"
            ],
            RejectionReason.TIME_CONCERN: [
                "建议寻找更快速、高效的方案",
                "可以并行施工以缩短工期"
            ],
            RejectionReason.SAFETY_CONCERN: [
                "建议寻找更安全可靠的方案",
                "需要加强安全防护措施"
            ],
            RejectionReason.ALREADY_TRIED: [
                "建议尝试不同的方案",
                "可以结合实际情况调整已尝试的方案"
            ],
            RejectionReason.NEED_MORE_INFO: [
                "可以询问更多关于方案细节的问题",
                "建议查看方案的历史案例和实施效果"
            ],
            RejectionReason.OTHER: [
                "请提供更具体的反馈",
                "可以描述期望的方案特点"
            ]
        }

        return suggestions_map.get(reason, ["请提供更多反馈信息"])

    def should_switch_plan(self, analysis: Dict[str, Any]) -> bool:
        """
        判断是否应该切换到备选方案

        Args:
            analysis: 分析结果

        Returns:
            是否切换方案
        """
        reason = analysis.get("reason")

        # 这些情况下应该切换方案
        switch_reasons = [
            RejectionReason.UNSUITABLE_CONDITION,
            RejectionReason.ALREADY_TRIED,
            RejectionReason.TOO_COMPLEX,
            RejectionReason.TOO_SIMPLE,
            RejectionReason.INSUFFICIENT_MEASURES
        ]

        return reason in switch_reasons

    def format_rejection_summary(self, analysis: Dict[str, Any]) -> str:
        """
        格式化拒绝原因总结

        Args:
            analysis: 分析结果

        Returns:
            格式化的总结字符串
        """
        reason = analysis.get("reason")
        suggestions = analysis.get("suggestions", [])

        summary_parts = [
            f"拒绝原因: {reason.value if isinstance(reason, RejectionReason) else reason}",
            "调整建议:"
        ]

        for i, suggestion in enumerate(suggestions, 1):
            summary_parts.append(f"  {i}. {suggestion}")

        return "\n".join(summary_parts)
