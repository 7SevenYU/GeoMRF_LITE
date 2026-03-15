import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI
from kg_construction.utils.logger import setup_logger


def _get_project_root():
    """获取项目根目录"""
    # 从 llm_client.py 向上4级到项目根目录
    # llm_client.py -> extraction -> core -> kg_construction -> 项目根目录
    return Path(__file__).parent.parent.parent.parent


class TokenLimitError(Exception):
    """当输入文本超过模型的token限制时抛出此异常"""
    pass


class LLMClient:
    def __init__(self, config_path: str = None):
        self.logger = setup_logger("LLMClient", "extraction.log")

        if config_path is None:
            config_path = "config/llm_config.json"

        # 将相对路径转换为绝对路径
        if not Path(config_path).is_absolute():
            config_path = _get_project_root() / config_path

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.current_model = self.config.get("current_model")
        if not self.current_model:
            raise ValueError("config.json must specify 'current_model'")
        self.client = self._init_client()

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Loaded LLM config from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise

    def _get_model_config(self) -> Dict[str, Any]:
        """获取当前模型的配置"""
        model_config = self.config.get("models", {}).get(self.current_model, {})
        if not model_config:
            raise ValueError(f"Model config not found for: {self.current_model}")
        return model_config

    def _init_client(self) -> OpenAI:
        model_config = self._get_model_config()

        api_base = model_config.get("api_base")
        api_key = model_config.get("api_key")

        if not api_base or not api_key:
            raise ValueError(f"Invalid API configuration for model {self.current_model}")

        return OpenAI(
            base_url=api_base,
            api_key=api_key
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        max_retries: int = 3
    ) -> Optional[str]:
        model_config = self._get_model_config()

        if model is None:
            model = model_config.get("model_name")
            if not model:
                raise ValueError(f"model_name not configured for {self.current_model}")
        if temperature is None:
            temperature = model_config.get("temperature", 0.7)
        if max_tokens is None:
            max_tokens = model_config.get("max_tokens", 2000)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                self.logger.info(f"LLM request succeeded on attempt {attempt + 1}")
                return content

            except Exception as e:
                error_msg = str(e).lower()
                # Check for token limit related errors
                if any(keyword in error_msg for keyword in ['token', 'length', 'limit', 'too long', 'maximum']):
                    self.logger.warning(f"Token limit exceeded on attempt {attempt + 1}: {e}")
                    raise TokenLimitError(f"Input text exceeds model token limit: {e}")

                self.logger.warning(f"LLM request failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self.logger.error("LLM request failed after all retries")
                    return None

    def batch_extract(
        self,
        prompts: List[str],
        system_prompt: str = None,
        **kwargs
    ) -> List[Optional[str]]:
        results = []

        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            result = self.chat(messages, **kwargs)
            results.append(result)

        return results

    def extract_json(
        self,
        prompt: str,
        system_prompt: str = None,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                content = self.chat(messages, max_retries=1)
                if content is None:
                    continue

                json_str = self._extract_json_from_response(content)
                if json_str:
                    return json.loads(json_str)

            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    continue

        return None

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        response = response.strip()

        if response.startswith("{") and response.endswith("}"):
            return response

        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx + 1]

        return None
