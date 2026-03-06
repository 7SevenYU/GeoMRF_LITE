import json
from pathlib import Path
from typing import Dict, List, Any
from kg_construction.core.extraction.base_extractor import BaseExtractor, ExtractionResult, ExtractedNode, ExtractedRelation
from kg_construction.core.extraction.config import NODE_TYPES, RELATION_TYPES, ExtractorType, SemanticExtractorType, LEXICON_CONFIG
from kg_construction.core.extraction.llm_client import LLMClient
from kg_construction.utils.logger import setup_logger


def _get_project_root():
    """获取项目根目录"""
    # 从 llm_extractor.py 向上4级到项目根目录
    # llm_extractor.py -> extraction -> core -> kg_construction -> 项目根目录
    return Path(__file__).parent.parent.parent.parent


# 常见中文关系类型到标准英文关系类型的映射
RELATION_TYPE_MAPPING = {
    "应对": "RESPONDS_TO",
    "针对": "RESPONDS_TO",
    "应对风险": "RESPONDS_TO",
    "处理": "RESPONDS_TO",
    "响应": "RESPONDS_TO",
    "对应": "RESPONDS_TO",
    "适用于": "RESPONDS_TO",
    "考虑": "CONSIDERS"
}


class LLMExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any], id_generator):
        super().__init__(config, id_generator)
        self.logger = setup_logger("LLMExtractor", "extraction.log")
        self.llm_client = LLMClient()
        self.prompts_config = self._load_prompts_config()
        self.invalid_nodes_stats = {
            "invalid_risk_types": [],
            "total_valid_nodes": 0,
            "total_invalid_nodes": 0
        }

    def _load_prompts_config(self) -> Dict[str, Any]:
        """加载提示词配置文件"""
        config_path = _get_project_root() / "kg_construction/configs/prompts_config.json"
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded prompts config from {config_path}")
                return config
            else:
                self.logger.warning(f"Prompts config file not found: {config_path}, using default prompts")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load prompts config: {e}, using default prompts")
            return {}

    def extract(self, text: str, **kwargs) -> ExtractionResult:
        chunk_id = kwargs.get("chunk_id", "")
        document_type = kwargs.get("document_type", "")
        source_file = kwargs.get("source_file", "")

        result = ExtractionResult(
            chunk_id=chunk_id,
            document_type=document_type,
            source_file=source_file
        )

        doc_config = self.config.get(document_type, {})
        nodes_config = doc_config.get("nodes", {})
        relations_config = doc_config.get("relations", {})

        node_prompt = self._build_node_extraction_prompt(document_type, nodes_config, text)
        system_prompt = self._get_system_prompt(document_type)

        extracted_data = self.llm_client.extract_json(
            prompt=node_prompt,
            system_prompt=system_prompt
        )

        if extracted_data:
            nodes = extracted_data.get("nodes", [])
            relations_data = extracted_data.get("relations", [])

            for node_data in nodes:
                node = self._build_node_from_llm_output(node_data, nodes_config)
                if node:
                    validated_node = self._validate_and_normalize_node(node)
                    if validated_node:
                        result.nodes.append(validated_node)

            for rel_data in relations_data:
                relation = self._build_relation_from_llm_output(rel_data, relations_config, result.nodes)
                if relation:
                    result.relations.append(relation)

        self.logger.info(f"Extracted {len(result.nodes)} nodes and {len(result.relations)} relations using LLM from chunk {chunk_id}")
        return result

    def _get_system_prompt(self, document_type: str = None) -> str:
        """从配置文件获取系统提示词，如果不存在则使用默认"""
        if self.prompts_config:
            system_prompts = self.prompts_config.get("system_prompts", {})
            return system_prompts.get("default", "你是一个专业的地质信息抽取专家。请严格按照JSON格式返回提取结果。")
        return "你是一个专业的地质信息抽取专家。请严格按照JSON格式返回提取结果。"

    def _build_node_extraction_prompt(self, document_type: str, nodes_config: Dict[str, Any], text: str) -> str:
        """从配置文件构建节点提取提示词"""
        if self.prompts_config and document_type in self.prompts_config.get("node_extraction_prompts", {}):
            return self._build_prompt_from_config(document_type, nodes_config, text)
        else:
            return self._build_default_prompt(nodes_config, text)

    def _build_prompt_from_config(self, document_type: str, nodes_config: Dict[str, Any], text: str) -> str:
        """使用prompts_config构建提示词"""
        doc_prompt_config = self.prompts_config["node_extraction_prompts"][document_type]
        instructions = doc_prompt_config.get("instructions", "")
        attribute_descriptions = doc_prompt_config.get("attribute_descriptions", {})
        examples = doc_prompt_config.get("examples", [])

        prompt = f"{instructions}\n\n"
        prompt += "【文本内容】\n"
        prompt += text
        prompt += "\n\n【需要提取的节点及其属性】\n"

        for node_name, node_config in nodes_config.items():
            required = node_config.get("required", False)
            attributes = node_config.get("attributes", [])
            description = node_config.get("description", "")

            prompt += f"\n{node_name}（{'必需' if required else '可选'}）\n"
            prompt += f"  描述：{description}\n"
            prompt += f"  属性：\n"

            for attr in attributes:
                attr_desc = attribute_descriptions.get(attr, f"提取{attr}信息")
                prompt += f"    - {attr}: {attr_desc}\n"

        if examples:
            prompt += "\n【提取示例】\n"
            for i, example in enumerate(examples, 1):
                prompt += f"\n示例{i}:\n"
                prompt += f"输入: {example['input']}\n"
                prompt += f"输出: {example['output']}\n"

        prompt += "\n\n请严格按照上述要求提取信息，并以JSON格式返回：\n"
        prompt += """{
    "nodes": [
        {
            "node_type": "节点类型名称",
            "attributes": {
                "属性1": "值1",
                "属性2": "值2"
            }
        }
    ],
    "relations": [
        {
            "head_type": "头节点类型",
            "tail_type": "尾节点类型",
            "relation_type": "关系类型"
        }
    ]
}"""

        return prompt

    def _build_default_prompt(self, nodes_config: Dict[str, Any], text: str) -> str:
        """构建默认提示词（向后兼容）"""
        prompt = "请从以下文本中提取地质信息节点和关系：\n\n"
        prompt += "【文本内容】\n"
        prompt += text
        prompt += "\n\n【需要提取的节点类型】\n"

        for node_name, node_config in nodes_config.items():
            required = node_config.get("required", False)
            attributes = node_config.get("attributes", [])
            description = node_config.get("description", "")

            prompt += f"\n{node_name}（{'必需' if required else '可选'}）：{description}\n"
            prompt += f"  - 属性：{', '.join(attributes)}\n"

        prompt += "\n\n请以JSON格式返回提取结果。"
        return prompt

    def _build_node_from_llm_output(
        self,
        node_data: Dict[str, Any],
        nodes_config: Dict[str, Any]
    ) -> ExtractedNode:
        try:
            node_type = node_data.get("node_type")
            attributes = node_data.get("attributes", {})

            if not node_type or node_type not in nodes_config:
                self.logger.warning(f"Invalid node type: {node_type}")
                return None

            node_config = nodes_config[node_type]
            node_type_config = NODE_TYPES.get(node_type, {})

            node_id = self.id_gen.generate_node_id()
            cypher_label = node_type_config.get("cypher_label", node_type)
            category = node_type_config.get("category", "S").value

            merge_keys = self._generate_merge_keys(cypher_label, attributes)

            node = ExtractedNode(
                node_id=node_id,
                node_type=node_type,
                node_label=node_type_config.get("label", node_type),
                cypher_label=cypher_label,
                attributes=attributes,
                merge_keys=merge_keys,
                category=category,
                confidence=0.85,
                extraction_method="llm"
            )

            return node

        except Exception as e:
            self.logger.error(f"Error building node from LLM output: {e}")
            return None

    def _validate_and_normalize_node(self, node: ExtractedNode) -> ExtractedNode:
        """验证和规范化节点属性"""
        if node.node_type == "风险类型":
            node = self._validate_risk_type(node)

        return node

    def _validate_risk_type(self, node: ExtractedNode) -> ExtractedNode:
        """验证风险类型是否在规范值中"""
        valid_risk_types = LEXICON_CONFIG.get("risk_type", {}).get("keywords", [])

        risk_type = node.attributes.get("riskType", "")
        if risk_type and risk_type not in valid_risk_types:
            # 记录无效的值及其来源
            self.invalid_nodes_stats["invalid_risk_types"].append({
                "invalid_value": risk_type,
                "valid_values": valid_risk_types,
                "source_file": getattr(node, 'source_file', 'unknown'),
                "node_id": node.node_id
            })
            self.invalid_nodes_stats["total_invalid_nodes"] += 1

            self.logger.warning(
                f"⚠️ Invalid risk type '{risk_type}' extracted from {getattr(node, 'source_file', 'unknown')}. "
                f"Valid values: {valid_risk_types}. "
                f"This node will be SKIPPED. Consider improving the prompt."
            )
            return None

        self.invalid_nodes_stats["total_valid_nodes"] += 1
        return node

    def print_validation_stats(self):
        """打印验证统计信息"""
        stats = self.invalid_nodes_stats
        if stats["total_invalid_nodes"] > 0:
            self.logger.warning("=" * 80)
            self.logger.warning("📊 Risk Type Validation Statistics")
            self.logger.warning("=" * 80)
            self.logger.warning(f"Total valid nodes: {stats['total_valid_nodes']}")
            self.logger.warning(f"Total invalid nodes: {stats['total_invalid_nodes']}")
            self.logger.warning(f"Invalid rate: {stats['total_invalid_nodes'] / (stats['total_valid_nodes'] + stats['total_invalid_nodes']) * 100:.1f}%")
            self.logger.warning("\nInvalid values extracted:")
            for item in stats["invalid_risk_types"]:
                self.logger.warning(f"  - '{item['invalid_value']}' from {item['source_file']}")
            self.logger.warning("\n💡 Consider improving the prompt if invalid rate > 5%")
            self.logger.warning("=" * 80)

    def _build_relation_from_llm_output(
        self,
        rel_data: Dict[str, Any],
        relations_config: Dict[str, Any],
        existing_nodes: List[ExtractedNode]
    ) -> ExtractedRelation:
        try:
            head_type = rel_data.get("head_type")
            tail_type = rel_data.get("tail_type")
            relation_type = rel_data.get("relation_type")

            if not all([head_type, tail_type, relation_type]):
                return None

            # 映射中文关系类型到标准英文关系类型
            original_relation_type = relation_type
            if relation_type in RELATION_TYPE_MAPPING:
                relation_type = RELATION_TYPE_MAPPING[relation_type]
                self.logger.info(f"Mapped relation type: '{original_relation_type}' -> '{relation_type}'")

            head_node = self._find_node_by_type(existing_nodes, head_type)
            tail_node = self._find_node_by_type(existing_nodes, tail_type)

            if not head_node or not tail_node:
                return None

            relation_id = self.id_gen.generate_relation_id()
            relation_config = RELATION_TYPES.get(relation_type, {})

            relation = ExtractedRelation(
                relation_id=relation_id,
                relation_type=relation_type,
                relation_label=relation_config.get("label", relation_type),
                cypher_label=relation_config.get("cypher_label", relation_type),
                head_node_id=head_node.node_id,
                tail_node_id=tail_node.node_id,
                head_merge_key=head_node.merge_keys[0] if head_node.merge_keys else "",
                tail_merge_key=tail_node.merge_keys[0] if tail_node.merge_keys else "",
                confidence=0.8,
                extraction_method="llm"
            )

            return relation

        except Exception as e:
            self.logger.error(f"Error building relation from LLM output: {e}")
            return None

    def extract_with_prompt(self, prompt: str, **kwargs) -> ExtractionResult:
        chunk_id = kwargs.get("chunk_id", "")
        document_type = kwargs.get("document_type", "")
        source_file = kwargs.get("source_file", "")

        result = ExtractionResult(
            chunk_id=chunk_id,
            document_type=document_type,
            source_file=source_file
        )

        doc_config = self.config.get(document_type, {})
        nodes_config = doc_config.get("nodes", {})
        relations_config = doc_config.get("relations", {})

        system_prompt = self._get_system_prompt(document_type)

        extracted_data = self.llm_client.extract_json(
            prompt=prompt,
            system_prompt=system_prompt
        )

        if extracted_data:
            attributes = extracted_data.get("attributes", {})

            nodes_data = extracted_data.get("nodes", [])
            if not nodes_data and attributes:
                default_node_type = list(nodes_config.keys())[0] if nodes_config else "Unknown"
                nodes_data = [{"node_type": default_node_type, "attributes": attributes}]

            for node_data in nodes_data:
                node = self._build_node_from_llm_output(node_data, nodes_config)
                if node:
                    result.nodes.append(node)

            relations_data = extracted_data.get("relations", [])
            for rel_data in relations_data:
                relation = self._build_relation_from_llm_output(rel_data, relations_config, result.nodes)
                if relation:
                    result.relations.append(relation)

        self.logger.info(f"Extracted {len(result.nodes)} nodes and {len(result.relations)} relations using custom prompt from chunk {chunk_id}")
        return result

    def _find_node_by_type(self, nodes: List[ExtractedNode], node_type: str) -> ExtractedNode:
        for node in nodes:
            if node.node_type == node_type:
                return node
        return None

    def _generate_merge_keys(self, cypher_label: str, attributes: Dict[str, Any]) -> List[str]:
        merge_keys = [cypher_label]
        for key, value in attributes.items():
            if value:
                merge_keys.append(f"{key}:{value}")
        return merge_keys

    def batch_extract(self, texts: List[str], **kwargs) -> List[ExtractionResult]:
        results = []
        document_type = kwargs.get("document_type", "")
        source_file = kwargs.get("source_file", "")

        doc_config = self.config.get(document_type, {})
        nodes_config = doc_config.get("nodes", {})

        prompts = []
        for i, text in enumerate(texts):
            prompt = self._build_node_extraction_prompt(nodes_config, text)
            prompts.append(prompt)

        system_prompt = self._get_system_prompt()
        responses = self.llm_client.batch_extract(prompts, system_prompt=system_prompt)

        for i, (text, response) in enumerate(zip(texts, responses)):
            chunk_id = kwargs.get("chunk_id", f"{i}")

            result = ExtractionResult(
                chunk_id=chunk_id,
                document_type=document_type,
                source_file=source_file
            )

            if response:
                try:
                    extracted_data = json.loads(response)
                    nodes = extracted_data.get("nodes", [])
                    relations_data = extracted_data.get("relations", [])

                    for node_data in nodes:
                        node = self._build_node_from_llm_output(node_data, nodes_config)
                        if node:
                            result.nodes.append(node)

                    for rel_data in relations_data:
                        relation = self._build_relation_from_llm_output(
                            rel_data,
                            doc_config.get("relations", {}),
                            result.nodes
                        )
                        if relation:
                            result.relations.append(relation)

                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse LLM response for chunk {chunk_id}: {e}")

            results.append(result)

        return results
