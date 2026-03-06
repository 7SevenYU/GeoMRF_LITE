"""
LLM客户端 - 支持OpenAI兼容API（Qwen等）
"""
import sys
import json
from pathlib import Path
from openai import OpenAI

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class LLMClient:
    """LLM客户端 - 支持OpenAI兼容API"""

    def __init__(self, config_path: str = None):
        """
        初始化LLM客户端

        Args:
            config_path: LLM配置文件路径，默认使用项目配置
        """
        if config_path is None:
            config_path = project_root / "config" / "llm_config.json"

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 获取当前模型配置
        current_model = config.get("current_model", "qwen")
        model_config = config["models"][current_model]

        self.client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["api_base"]
        )
        self.model_name = model_config["model_name"]
        self.default_temperature = model_config.get("temperature", 0.7)
        self.default_max_tokens = model_config.get("max_tokens", 2000)

    def generate(self, messages: list, **kwargs) -> str:
        """
        生成LLM回复

        Args:
            messages: 消息列表
            **kwargs: 额外参数（temperature, max_tokens等）

        Returns:
            LLM生成的文本回复
        """
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()


if __name__ == '__main__':
    # 测试LLM客户端
    client = LLMClient()
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，请介绍一下TBM隧道施工。"}
    ]
    response = client.generate(messages)
    print(f"LLM回复: {response}")
