import json
from pathlib import Path
from typing import Dict, Optional, List
from kg_construction.utils.logger import setup_logger


def _get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent


class PromptLoader:
    """Prompt加载器 - 支持命名空间"""

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            project_root = _get_project_root()
            base_path = project_root / "kg_construction" / "prompts"
        self.base_path = Path(base_path)
        self.logger = setup_logger("PromptLoader", "extraction.log")

        self.mapping = self._load_mapping()

    def _load_mapping(self) -> Dict:
        mapping_file = self.base_path / "config" / "prompts_mapping.json"
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load prompt mapping: {e}")
            return {}

    def get_document_config(self, document_type: str) -> Dict:
        """获取文档的完整配置"""
        return self.mapping.get("document_types", {}).get(document_type, {})

    def get_section_prompt(self, document_type: str, section_id: str, text: str) -> Optional[str]:
        """获取特定章节的prompt（支持命名空间）"""
        if not document_type:
            self.logger.warning("Document type is empty")
            return None

        doc_config = self.get_document_config(document_type)
        segmentation = doc_config.get("segmentation", {})
        namespace = segmentation.get("namespace")

        if not namespace:
            self.logger.warning(f"No namespace configured for document type {document_type}")
            return None

        sections = doc_config.get("sections", {})

        if section_id not in sections:
            self.logger.warning(f"No prompt config for section {section_id} in document type {document_type}")
            return None

        section_config = sections[section_id]
        prompt_file = section_config.get("prompt_file")

        if not prompt_file:
            return None

        prompt_path = self.base_path / namespace / prompt_file
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()

            prompt = prompt_template.replace("{text}", text)

            self.logger.info(f"Loaded prompt for {document_type}/section{section_id}: {prompt_file}")
            return prompt

        except Exception as e:
            self.logger.error(f"Failed to load prompt file {prompt_path}: {e}")
            return None

    def get_system_prompt(self) -> str:
        system_prompt_file = self.base_path / "common" / "system_prompt.txt"
        try:
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load system prompt: {e}")
            return "你是地质信息提取专家。"

    def get_section_config(self, document_type: str, section_id: str) -> Optional[Dict]:
        """获取章节配置"""
        if not document_type:
            return None

        doc_config = self.get_document_config(document_type)
        sections = doc_config.get("sections", {})

        return sections.get(section_id)

    def get_document_sections(self, document_type: str) -> List[str]:
        """获取文档的所有有效章节编号"""
        doc_config = self.get_document_config(document_type)
        sections = doc_config.get("sections", {})

        valid_sections = []
        for section_num, config in sections.items():
            if config.get("extraction_method") != "none":
                valid_sections.append(section_num)

        return sorted(valid_sections)

    def get_extraction_method(self, document_type: str, section_id: str) -> str:
        """获取章节的提取方法"""
        section_config = self.get_section_config(document_type, section_id)
        if section_config:
            return section_config.get("extraction_method", "none")
        return "none"
