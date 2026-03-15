"""
对话管理器 - 重构版

集成新组件：
- RecommendationStateMachine: 状态机管理
- ResponseGenerator: 提示词生成
- FeedbackAnalyzer: 反馈分析
- IntentionPredictor: 意图识别
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.core.recommendation_engine import RecommendationEngine
from recommendation.models.llm.llm_client import LLMClient
from recommendation.models.intention.intention_predict import IntentionPredictor

from recommendation.core.state_machine import (
    RecommendationStateMachine,
    RecommendationState,
    QueryContext,
    PlanResult
)
from recommendation.core.response_generator import ResponseGenerator
from recommendation.core.feedback_analyzer import FeedbackAnalyzer

from retrieval.utils.kg_utils import extract_key_spa, extract_key_risk
from recommendation.utils import config

logger = logging.getLogger(__name__)


class TBMRiskConversationManager:
    """TBM风险对话管理器 - 重构版"""

    def __init__(self, prompt_config_path: Optional[str] = None):
        """
        初始化对话管理器

        Args:
            prompt_config_path: 提示词配置文件路径
        """
        # 核心组件
        self.engine = RecommendationEngine()
        self.state_machine = RecommendationStateMachine(max_rejections=config.MAX_REJECTIONS)
        self.response_generator = ResponseGenerator(prompt_config_path)
        self.feedback_analyzer = FeedbackAnalyzer()
        self.intention_predictor = IntentionPredictor()
        self.llm_client = LLMClient()

        logger.info("对话管理器初始化完成（重构版）")

    def chat(self, query: str, **llm_kwargs) -> str:
        """
        处理用户查询

        Args:
            query: 用户查询文本
            **llm_kwargs: LLM参数（temperature, max_tokens等）

        Returns:
            系统回复
        """
        query = query.strip()
        if not query:
            return "输入不能为空，请您描述清楚风险类型或地质情况。"

        logger.info(f"收到查询: {query[:50]}...")
        logger.debug(f"当前状态: {self.state_machine.get_status_summary()}")

        # 根据状态机处理查询
        if self.state_machine.state == RecommendationState.IDLE:
            return self._handle_new_query(query, **llm_kwargs)

        elif self.state_machine.state == RecommendationState.PRESENTING:
            return self._handle_presenting_state(query, **llm_kwargs)

        elif self.state_machine.state == RecommendationState.REFUSED:
            return self._handle_refused_state(query, **llm_kwargs)

        elif self.state_machine.state == RecommendationState.COMPLETED:
            # 对话已完成，重置并处理新查询
            self.state_machine.reset()
            return self._handle_new_query(query, **llm_kwargs)

        else:
            logger.warning(f"未知状态: {self.state_machine.state}")
            return "系统状态异常，请重新开始。"

    def _handle_new_query(self, query: str, **llm_kwargs) -> str:
        """
        处理新查询

        Args:
            query: 用户查询
            **llm_kwargs: LLM参数

        Returns:
            系统回复
        """
        # 1. 意图识别（在检索前执行，节省token）
        is_relevant = self.intention_predictor.classify(query)
        if not is_relevant:
            logger.info("意图识别：非处置相关查询")
            return "请问您有什么关于TBM地质风险防控的问题？"

        logger.info("意图识别：处置相关查询，开始检索")

        # 2. 提取关键信息
        key_spa, line_name = extract_key_spa(query)
        risk_type = extract_key_risk(query)

        # 提取NER实体
        try:
            geo_entities = self.engine.search_engine.ner_predictor.predict(query)
        except Exception as e:
            logger.error(f"NER预测失败: {e}")
            geo_entities = []

        # 调试打印：直接输出到控制台
        print(f"[DEBUG] 原始查询: {query}")
        print(f"[DEBUG] 提取结果: key_spa={key_spa}, line_name={line_name}, risk_type={risk_type}")
        print(f"[DEBUG] NER实体: {geo_entities}")
        logger.info(f"提取结果: key_spa={key_spa}, line_name={line_name}, risk_type={risk_type}, NER实体={geo_entities}")

        # 3. 启动状态机检索流程
        self.state_machine.start_search(
            query=query,
            line_name=line_name,
            mileage=key_spa,
            risk_type=risk_type,
            tbm_name="XX隧道",
            geo_entities=geo_entities
        )

        # 4. 执行检索
        success = self.engine.start_recommendation(query)
        if not success:
            self.state_machine.reset()
            return "未找到推荐方案，请补充更多地质信息或风险描述。"

        # 5. 获取检索结果并设置到状态机
        engine_results = self.engine.state["results"]

        # 调试打印：直接输出检索结果
        print(f"[DEBUG] 检索结果数量: {len(engine_results)}")
        if engine_results:
            for idx, result in enumerate(engine_results):
                node_id = result.get('node_id', 'N/A')
                plan_keys = list(result.get('plan_data', {}).keys())
                design_keys = list(result.get('design_data', {}).keys())
                print(f"[DEBUG] 方案{idx+1}: node_id={node_id}")
                print(f"[DEBUG]   plan_data字段数: {len(plan_keys)}, 字段: {plan_keys}")
                print(f"[DEBUG]   design_data字段数: {len(design_keys)}, 字段: {design_keys}")

        # 调试日志：打印检索结果摘要
        if engine_results:
            first_result = engine_results[0]
            logger.info(f"检索结果: plan_data keys={list(first_result.get('plan_data', {}).keys())}, "
                       f"design_data keys={list(first_result.get('design_data', {}).keys())}")

        plan_results = []
        for idx, result in enumerate(engine_results):
            # 分离当前数据和历史数据
            plan_data = result.get("plan_data", {})
            design_data = result.get("design_data", {})

            # 创建合并的数据字典，明确区分当前和历史
            merged_plan_data = {}

            # 当前数据（里程关联检索）- 数据已包含current_前缀
            if design_data:
                merged_plan_data["current_design_info"] = design_data.get("current_design_info", {})
                merged_plan_data["current_construction_info"] = design_data.get("current_construction_info", {})
                merged_plan_data["current_detection_methods"] = design_data.get("current_detection_methods", [])
                merged_plan_data["current_detection_conclusions"] = design_data.get("current_detection_conclusions", [])
                merged_plan_data["current_geological_risk_levels"] = design_data.get("current_geological_risk_levels", [])
                merged_plan_data["current_risk_assessments"] = design_data.get("current_risk_assessments", [])

            # 历史数据（方案关联检索）- 保持原有字段名
            merged_plan_data.update(plan_data)

            plan_result = PlanResult(
                plan_id=idx,  # 使用索引作为临时ID
                plan_data=merged_plan_data,
                rank=idx + 1
            )
            plan_results.append(plan_result)

        self.state_machine.set_results(plan_results)

        # 6. 生成首次推荐回复
        return self._generate_first_recommendation(**llm_kwargs)

    def _handle_presenting_state(self, query: str, **llm_kwargs) -> str:
        """
        处理方案展示状态下的查询

        增加新问题检测：如果NER、里程、风险类型、线路任一发生变化，
        则认为是新问题，重置状态机并重新检索

        Args:
            query: 用户查询
            **llm_kwargs: LLM参数

        Returns:
            系统回复
        """
        # 1. 提取新查询的关键信息
        new_key_spa, new_line_name = extract_key_spa(query)
        new_risk_type = extract_key_risk(query)

        # 2. 提取新查询的NER实体
        try:
            new_geo_entities = self.engine.search_engine.ner_predictor.predict(query)
        except Exception as e:
            logger.error(f"NER预测失败: {e}")
            new_geo_entities = []

        # 3. 获取原查询的信息
        old_context = self.state_machine.context
        old_mileage = old_context.mileage if old_context else None
        old_risk_type = old_context.risk_type if old_context else None
        old_line_name = old_context.line_name if old_context else None
        old_geo_entities = old_context.geo_entities if old_context else []

        # 4. 判断是否为新问题（任一维度变化即认为是新问题）
        is_new_problem = (
            new_key_spa != old_mileage or           # 里程变化
            new_risk_type != old_risk_type or       # 风险类型变化
            new_line_name != old_line_name or       # 线路变化
            set(new_geo_entities) != set(old_geo_entities)  # NER实体变化
        )

        # 5. 如果检测到新问题，重置状态机并开始新检索
        if is_new_problem:
            print(f"[DEBUG] 检测到新问题，重置状态机并开始新检索")
            print(f"[DEBUG]   里程: {old_mileage} -> {new_key_spa}")
            print(f"[DEBUG]   风险类型: {old_risk_type} -> {new_risk_type}")
            print(f"[DEBUG]   线路: {old_line_name} -> {new_line_name}")
            print(f"[DEBUG]   NER实体: {old_geo_entities} -> {new_geo_entities}")
            logger.info(f"检测到新问题，重置状态机并开始新检索")
            logger.info(f"  里程: {old_mileage} -> {new_key_spa}")
            logger.info(f"  风险类型: {old_risk_type} -> {new_risk_type}")
            logger.info(f"  线路: {old_line_name} -> {new_line_name}")
            logger.info(f"  NER实体: {old_geo_entities} -> {new_geo_entities}")

            self.state_machine.reset()
            return self._handle_new_query(query, **llm_kwargs)

        # 6. 否则按原有逻辑处理（拒绝/追问）
        # 判断是否是拒绝
        is_rejection = self._is_rejection_feedback(query)

        if is_rejection:
            # 拒绝当前方案
            analysis = self.feedback_analyzer.analyze_rejection(query)
            self.state_machine.reject_plan(query)

            if self.state_machine.is_completed:
                # 达到最大拒绝次数或无更多方案
                return self._generate_conversation_end(**llm_kwargs)
            else:
                # 切换到下一个方案
                return self._generate_refusal_handling(analysis, **llm_kwargs)
        else:
            # 追问或补充信息
            return self._generate_follow_up_response(query, **llm_kwargs)

    def _handle_refused_state(self, query: str, **llm_kwargs) -> str:
        """
        处理拒绝状态下的查询

        Args:
            query: 用户查询
            **llm_kwargs: LLM参数

        Returns:
            系统回复
        """
        # 从拒绝状态转换回展示状态
        self.state_machine._transition_to(RecommendationState.PRESENTING)
        return self._generate_first_recommendation(**llm_kwargs)

    def _is_rejection_feedback(self, query: str) -> bool:
        """
        判断是否是拒绝反馈

        Args:
            query: 用户查询

        Returns:
            是否拒绝
        """
        rejection_keywords = ["不满意", "不推荐", "不行", "不好", "换一个", "其他", "别的"]
        return any(keyword in query for keyword in rejection_keywords)

    def _generate_first_recommendation(self, **llm_kwargs) -> str:
        """
        生成首次推荐回复

        Args:
            **llm_kwargs: LLM参数

        Returns:
            生成的回复
        """
        current_plan = self.state_machine.current_plan
        if not current_plan:
            return "无可用方案"

        # 准备变量
        variables = self._prepare_scenario_variables(
            ResponseGenerator.SCENARIO_FIRST_RECOMMENDATION
        )

        # 生成提示词
        prompt = self.response_generator.generate_prompt(
            ResponseGenerator.SCENARIO_FIRST_RECOMMENDATION,
            variables
        )

        # 调用LLM生成回复
        messages = [
            {"role": "system", "content": "你是一个熟悉盾构施工（TBM）风险防控的工程专家。"},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.generate(messages, **llm_kwargs)
        logger.info("生成首次推荐回复")
        return response

    def _generate_refusal_handling(self, analysis: Dict[str, Any], **llm_kwargs) -> str:
        """
        生成拒绝处理回复

        Args:
            analysis: 反馈分析结果
            **llm_kwargs: LLM参数

        Returns:
            生成的回复
        """
        # 准备变量
        variables = self._prepare_scenario_variables(
            ResponseGenerator.SCENARIO_REFUSAL_HANDLING
        )
        variables["rejection_reason"] = analysis.get("keyword", "")
        variables["current_plan"] = self.state_machine.current_plan.plan_data

        # 生成提示词
        prompt = self.response_generator.generate_prompt(
            ResponseGenerator.SCENARIO_REFUSAL_HANDLING,
            variables
        )

        # 调用LLM生成回复
        messages = [
            {"role": "system", "content": "你是一个熟悉盾构施工（TBM）风险防控的工程专家。"},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.generate(messages, **llm_kwargs)

        # 添加建议
        suggestions = analysis.get("suggestions", [])
        if suggestions:
            response += "\n\n" + "\n".join(f"- {s}" for s in suggestions)

        logger.info("生成拒绝处理回复")
        return response

    def _generate_follow_up_response(self, query: str, **llm_kwargs) -> str:
        """
        生成追问回复

        Args:
            query: 用户问题
            **llm_kwargs: LLM参数

        Returns:
            生成的回复
        """
        # 准备变量
        variables = self._prepare_scenario_variables(
            ResponseGenerator.SCENARIO_FOLLOW_UP_QUESTION
        )
        variables["query"] = query

        # 生成提示词
        prompt = self.response_generator.generate_prompt(
            ResponseGenerator.SCENARIO_FOLLOW_UP_QUESTION,
            variables
        )

        # 调用LLM生成回复
        messages = [
            {"role": "system", "content": "你是一个熟悉盾构施工（TBM）风险防控的工程专家。"},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.generate(messages, **llm_kwargs)
        logger.info("生成追问回复")
        return response

    def _generate_conversation_end(self, **llm_kwargs) -> str:
        """
        生成对话结束回复

        Args:
            **llm_kwargs: LLM参数

        Returns:
            生成的回复
        """
        # 准备变量
        variables = self._prepare_scenario_variables(
            ResponseGenerator.SCENARIO_CONVERSATION_END
        )

        # 生成提示词
        prompt = self.response_generator.generate_prompt(
            ResponseGenerator.SCENARIO_CONVERSATION_END,
            variables
        )

        # 调用LLM生成回复
        messages = [
            {"role": "system", "content": "你是一个熟悉盾构施工（TBM）风险防控的工程专家。"},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.generate(messages, **llm_kwargs)
        logger.info("生成对话结束回复")
        return response

    def _prepare_scenario_variables(self, scenario: str) -> Dict[str, Any]:
        """
        为指定场景准备变量

        Args:
            scenario: 场景名称

        Returns:
            变量字典
        """
        context = self.state_machine.context
        current_plan = self.state_machine.current_plan

        variables = {
            "user_query": context.user_query if context else "",
            "query_params": {
                "tbm_name": context.tbm_name if context else "XX隧道",
                "line_name": context.line_name if context else "",
                "mileage": context.mileage if context else "",
                "risk_type": context.risk_type if context else ""
            },
            "plan_data": current_plan.plan_data if current_plan else {},
            "current_plan": current_plan.plan_data if current_plan else {}
        }

        return variables

    def reset(self):
        """重置对话状态"""
        self.state_machine.reset()
        self.engine.reset()
        logger.info("对话状态已重置")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    manager = TBMRiskConversationManager()

    # 测试对话
    test_query = "IV级深埋硬质岩掉块"
    print(f"测试查询: {test_query}")
    response = manager.chat(test_query)
    print(f"系统回复:\n{response}")
