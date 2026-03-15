"""
ResponseGenerator - 基于配置化提示词生成回复

功能：
1. 加载 response_prompts.json 配置
2. 根据场景选择对应的提示词模板
3. 填充变量到模板
4. 生成完整的 LLM 提示词
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    响应生成器

    基于配置化的提示词模板，根据不同场景生成 LLM 提示词
    """

    # 场景枚举
    SCENARIO_FIRST_RECOMMENDATION = "first_recommendation"
    SCENARIO_REFUSAL_HANDLING = "refusal_handling"
    SCENARIO_FOLLOW_UP_QUESTION = "follow_up_question"
    SCENARIO_SUPPLEMENTARY_INFO = "supplementary_info"
    SCENARIO_NO_RESULTS = "no_results"
    SCENARIO_CONVERSATION_END = "conversation_end"

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化响应生成器

        Args:
            config_path: 提示词配置文件路径
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "recommendation" / "data" / "response_prompts.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        logger.info(f"响应生成器初始化: config={self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"配置加载成功: {len(config['scenarios'])} 个场景")
            return config
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise

    def generate_prompt(
        self,
        scenario: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        生成提示词

        Args:
            scenario: 场景名称
            variables: 变量字典

        Returns:
            完整的提示词字符串
        """
        if scenario not in self.config["scenarios"]:
            logger.error(f"未知场景: {scenario}")
            return ""

        scenario_config = self.config["scenarios"][scenario]
        template_sections = scenario_config["template"]

        prompt_parts = []
        for section in template_sections:
            content = self._fill_variables(section["content"], variables)
            prompt_parts.append(content)

        full_prompt = "\n".join(prompt_parts)
        logger.debug(f"生成提示词: scenario={scenario}, length={len(full_prompt)}")
        return full_prompt

    def _fill_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """
        填充变量到模板

        Args:
            template: 模板字符串
            variables: 变量字典

        Returns:
            填充后的字符串
        """
        # 使用变量映射配置
        variable_mappings = self.config.get("variable_mappings", {})

        try:
            # 对于每个占位符 {variable_name}
            import re
            pattern = r'\{([^}]+)\}'

            def replace_var(match):
                var_name = match.group(1)
                # 查找变量映射
                mapped_path = variable_mappings.get(var_name, var_name)
                value = self._get_value_by_path(variables, mapped_path)
                # 处理None值：返回空字符串而不是占位符
                if value is None:
                    return ""
                return str(value)

            result = re.sub(pattern, replace_var, template)
            return result
        except Exception as e:
            logger.error(f"变量填充失败: {e}")
            return template

    def _get_value_by_path(self, data: Dict[str, Any], path: str) -> Any:
        """
        根据路径获取值

        支持嵌套访问和数组索引，如：
        - "plan_data.scheme.适用条件"
        - "design_data.current_detection_methods[0].detectionMethod"

        Args:
            data: 数据字典
            path: 点分隔的路径，支持数组索引

        Returns:
            找到的值，未找到返回 None
        """
        try:
            import re
            # 分割路径，但保留数组索引
            # 例如: "design_data.current_detection_methods[0].detectionMethod"
            # 分割为: ["design_data", "current_detection_methods[0]", "detectionMethod"]
            keys = path.split(".")
            value = data

            for key in keys:
                if value is None:
                    return None

                # 检查是否包含数组索引，如 "current_detection_methods[0]"
                match = re.match(r'([^\[]+)\[(\d+)\]', key)
                if match:
                    # 有数组索引
                    field_name = match.group(1)  # "current_detection_methods"
                    index = int(match.group(2))   # 0

                    if isinstance(value, dict):
                        value = value.get(field_name)
                        if isinstance(value, list) and 0 <= index < len(value):
                            value = value[index]
                        else:
                            return None
                    else:
                        return None
                else:
                    # 普通字段访问
                    if isinstance(value, dict):
                        value = value.get(key)
                    else:
                        return None

                if value is None:
                    return None

            return value
        except Exception as e:
            logger.debug(f"路径访问失败: path={path}, error={e}")
            return None

    def get_available_scenarios(self) -> List[str]:
        """获取所有可用场景"""
        return list(self.config["scenarios"].keys())

    def get_scenario_description(self, scenario: str) -> str:
        """获取场景描述"""
        if scenario in self.config["scenarios"]:
            return self.config["scenarios"][scenario].get("description", "")
        return ""
