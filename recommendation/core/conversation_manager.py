"""
对话管理器 - 使用prompt模板
"""
import sys
import json
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from recommendation.core.recommendation_engine import RecommendationEngine
from recommendation.core.feedback_handler import FeedbackHandler
from recommendation.models.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)


class TBMRiskConversationManager:
    """TBM风险对话管理器"""

    def __init__(self, prompt_template_path: str = None):
        """
        初始化对话管理器

        Args:
            prompt_template_path: prompt模板文件路径
        """
        self.engine = RecommendationEngine()
        self.feedback_handler = FeedbackHandler()
        self.llm_client = LLMClient()

        # 加载prompt模板
        if prompt_template_path is None:
            prompt_template_path = project_root / "recommendation" / "data" / "prompt_template.json"

        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = json.load(f)

        logger.info("对话管理器初始化完成")

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

        # 1. 如果已在推荐流程中，优先判断是否是拒绝
        if self.engine.state["active"]:
            if self.feedback_handler.is_rejection(query):
                return self._handle_rejection()

            # 判断是否是新提问
            if query != self.engine.state["first_query"] and self.engine.should_recommend(query):
                self.engine.reset()
                return self.chat(query, **llm_kwargs)

            # 重复查询，重置
            if query == self.engine.state["first_query"]:
                self.engine.reset()
                return self.chat(query, **llm_kwargs)

        # 2. 意图识别并开始推荐
        if not self.engine.should_recommend(query):
            return "请问您有什么关于TBM地质风险防控的问题？"

        if not self.engine.start_recommendation(query):
            self.feedback_handler.save_feedback(query, [])
            return "未找到推荐方案，请补充更多地质信息或风险描述。"

        # 3. 获取当前推荐并生成回复
        result = self.engine.get_current_recommendation()
        return self._generate_response(result, **llm_kwargs)

    def _handle_rejection(self) -> str:
        """处理拒绝反馈"""
        count = self.engine.increment_rejection_count()

        if count >= 3:
            first_q = self.engine.state["first_query"]
            results = self.engine.state["results"]
            self.feedback_handler.save_feedback(first_q, results, count)
            self.engine.reset()
            return f"您似乎对当前推荐方案都不满意，我已记录您的提问『{first_q}』和所有推荐方案，稍后反馈给工程师处理。"

        next_rec = self.engine.next_recommendation()
        if next_rec:
            return self._generate_response(next_rec)

        return "已无更多备选方案，请稍后再试或联系我们支持团队。"

    def _generate_response(self, result: dict, **llm_kwargs) -> str:
        """使用LLM生成自然语言回复"""
        prompt = self._build_prompt_from_template(result)
        messages = [
            {"role": "system", "content": "你是一个熟悉盾构施工（TBM）风险防控的工程专家。"},
            {"role": "user", "content": prompt}
        ]
        return self.llm_client.generate(messages, **llm_kwargs)

    def _build_prompt_from_template(self, result: dict) -> str:
        """使用模板文件构建prompt"""
        plan_data = result.get("plan_data", {})
        design_data = result.get("design_data", {})
        extracted_info = result.get("extracted_info", {})

        # 提取关联风险信息
        related_risk = plan_data.get("关联风险", {})
        history_design_info = related_risk.get("发生位置", {}).get("设计信息", {})
        history_detection_info = related_risk.get("发生位置", {}).get("探测信息", {})

        # 准备模板变量
        prompt_parts = {
            "query": extracted_info.get("key_spa_mileage", "未知"),
            "key_spa_line_name": extracted_info.get("key_spa_line", "未知"),
            "key_spa_mileage": extracted_info.get("key_spa_mileage", "未知"),
            "risk_type": extracted_info.get("key_risk", "未知"),
            "key_geo": extracted_info.get("key_geo", "未知"),

            "design_risk_eval": str(design_data.get("设计信息", {})),
            "design_detection_way": design_data.get("探测信息", {}).get("detectionMethod", "未知"),
            "design_conclusion": design_data.get("探测信息", {}).get("detectionConclusion", "未知"),
            "geological_risk_level": design_data.get("探测信息", {}).get("地质风险等级", "未知"),

            "applicable_conditions": plan_data.get("方案", {}).get("applicableConditions", "未知"),
            "plan": plan_data.get("方案", {}).get("emergencyResponseGuidelines", "未知"),

            "history_rock_surrounding_level": related_risk.get("发生位置", {}).get("设计信息", {}).get("围岩等级", "未知"),
            "early_warning_level": related_risk.get("预警等级", "未知"),
            "history_design_info": str(history_design_info),
            "history_detection_way": history_detection_info.get("detectionMethod", "未知"),
            "history_detection_conclusion": history_detection_info.get("detectionConclusion", "未知"),
            "history_geological_risk_level": history_detection_info.get("地质风险等级", "未知")
        }

        # 按模板顺序拼接
        full_prompt = ""
        for key in ["任务说明", "突发风险场景", "地质背景", "防控方案", "历史风险", "输出要求"]:
            template = self.prompt_template.get(key, "")
            try:
                filled = template.format(**prompt_parts)
            except KeyError as e:
                logger.warning(f"模板填充失败: {key}, 缺少键: {e}")
                filled = template
            full_prompt += filled + "\n"

        return full_prompt

    def reset(self):
        """重置对话状态"""
        self.engine.reset()
        logger.info("对话状态已重置")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    manager = TBMRiskConversationManager()

    # 测试对话
    test_query = "IV级深埋硬质岩掉块"
    print(f"测试查询: {test_query}")
    response = manager.chat(test_query)
    print(f"系统回复:\n{response}")
