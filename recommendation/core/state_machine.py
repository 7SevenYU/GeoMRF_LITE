"""
RecommendationStateMachine - 推荐状态机

管理推荐流程的状态转换：
IDLE → SEARCHING → PRESENTING → REFUSED → COMPLETED
                    ↓           ↓
                 COMPLETED    (切换方案)
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RecommendationState(Enum):
    """推荐状态枚举"""
    IDLE = "idle"                      # 空闲状态，等待查询
    SEARCHING = "searching"            # 检索中
    PRESENTING = "presenting"          # 展示方案
    REFUSED = "refused"                # 用户拒绝
    COMPLETED = "completed"            # 完成


@dataclass
class QueryContext:
    """查询上下文"""
    user_query: str                    # 用户原始查询
    line_name: Optional[str] = None    # 线路名称
    mileage: Optional[str] = None      # 里程
    risk_type: Optional[str] = None    # 风险类型
    tbm_name: str = "XX隧道"           # 隧道名称
    geo_entities: List[str] = None     # NER实体列表


@dataclass
class PlanResult:
    """方案结果"""
    plan_id: int                       # 方案ID
    plan_data: Dict[str, Any]          # 方案详细数据
    rank: int = 0                      # 排名


class RecommendationStateMachine:
    """
    推荐状态机

    功能：
    1. 管理推荐流程的状态转换
    2. 跟踪当前方案和备选方案列表
    3. 记录拒绝次数和原因
    4. 提供状态查询接口
    """

    def __init__(self, max_rejections: int = 3):
        """
        初始化状态机

        Args:
            max_rejections: 最大拒绝次数
        """
        self.max_rejections = max_rejections

        self.state = RecommendationState.IDLE
        self.context: Optional[QueryContext] = None
        self.results: List[PlanResult] = []
        self.current_index: int = 0
        self.rejection_count: int = 0
        self.rejection_reasons: List[str] = []

        logger.info(f"状态机初始化: 最大拒绝次数={max_rejections}")

    def start_search(self, query: str, line_name: Optional[str] = None,
                    mileage: Optional[str] = None, risk_type: Optional[str] = None,
                    tbm_name: str = "XX隧道", geo_entities: List[str] = None) -> None:
        """
        开始检索流程

        Args:
            query: 用户查询
            line_name: 线路名称
            mileage: 里程
            risk_type: 风险类型
            tbm_name: 隧道名称
            geo_entities: NER实体列表
        """
        if self.state != RecommendationState.IDLE:
            logger.warning(f"当前状态不为IDLE，无法开始检索: {self.state}")
            return

        self.context = QueryContext(
            user_query=query,
            line_name=line_name,
            mileage=mileage,
            risk_type=risk_type,
            tbm_name=tbm_name,
            geo_entities=geo_entities if geo_entities is not None else []
        )

        self.results = []
        self.current_index = 0
        self.rejection_count = 0
        self.rejection_reasons = []

        self._transition_to(RecommendationState.SEARCHING)
        logger.info(f"开始检索: query={query[:50]}, line={line_name}, mileage={mileage}")

    def set_results(self, results: List[PlanResult]) -> None:
        """
        设置检索结果

        Args:
            results: 方案结果列表
        """
        if self.state != RecommendationState.SEARCHING:
            logger.warning(f"当前状态不为SEARCHING，无法设置结果: {self.state}")
            return

        self.results = results
        self.current_index = 0

        if results:
            self._transition_to(RecommendationState.PRESENTING)
            logger.info(f"设置结果: 共{len(results)}个方案，当前展示第1个")
        else:
            logger.warning("无检索结果")
            self._transition_to(RecommendationState.COMPLETED)

    def accept_plan(self) -> None:
        """接受当前方案"""
        if self.state != RecommendationState.PRESENTING:
            logger.warning(f"当前状态不为PRESENTING，无法接受方案: {self.state}")
            return

        self._transition_to(RecommendationState.COMPLETED)
        logger.info(f"用户接受方案: 方案ID={self.current_plan.plan_id}")

    def reject_plan(self, reason: str = "") -> None:
        """
        拒绝当前方案

        Args:
            reason: 拒绝原因
        """
        if self.state != RecommendationState.PRESENTING:
            logger.warning(f"当前状态不为PRESENTING，无法拒绝方案: {self.state}")
            return

        self.rejection_count += 1
        self.rejection_reasons.append(reason)

        logger.info(f"用户拒绝方案: 方案ID={self.current_plan.plan_id}, "
                   f"原因={reason}, 拒绝次数={self.rejection_count}/{self.max_rejections}")

        if self.rejection_count >= self.max_rejections:
            logger.warning("达到最大拒绝次数，结束推荐")
            self._transition_to(RecommendationState.COMPLETED)
        elif self.current_index < len(self.results) - 1:
            self.current_index += 1
            self._transition_to(RecommendationState.REFUSED)
            logger.info(f"切换到下一个方案: 第{self.current_index + 1}个")
        else:
            logger.warning("没有更多备选方案")
            self._transition_to(RecommendationState.COMPLETED)

    def reset(self) -> None:
        """重置状态机"""
        self.state = RecommendationState.IDLE
        self.context = None
        self.results = []
        self.current_index = 0
        self.rejection_count = 0
        self.rejection_reasons = []
        logger.info("状态机已重置")

    def _transition_to(self, new_state: RecommendationState) -> None:
        """状态转换"""
        old_state = self.state
        self.state = new_state
        logger.debug(f"状态转换: {old_state} -> {new_state}")

    @property
    def current_plan(self) -> Optional[PlanResult]:
        """获取当前方案"""
        if 0 <= self.current_index < len(self.results):
            return self.results[self.current_index]
        return None

    @property
    def has_more_plans(self) -> bool:
        """是否还有更多备选方案"""
        return self.current_index < len(self.results) - 1

    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.state == RecommendationState.COMPLETED

    @property
    def is_presenting(self) -> bool:
        """是否正在展示方案"""
        return self.state == RecommendationState.PRESENTING

    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "state": self.state.value,
            "has_context": self.context is not None,
            "total_results": len(self.results),
            "current_index": self.current_index,
            "rejection_count": self.rejection_count,
            "max_rejections": self.max_rejections,
            "has_more_plans": self.has_more_plans
        }
