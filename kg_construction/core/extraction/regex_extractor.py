import re
from typing import Dict, List, Any
from kg_construction.core.extraction.base_extractor import BaseExtractor, ExtractionResult, ExtractedNode
from kg_construction.core.extraction.config import REGEX_PATTERNS, NODE_TYPES, ExtractorType, NodeCategory
from kg_construction.utils.logger import setup_logger


class RegexExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any], id_generator):
        super().__init__(config, id_generator)
        self.logger = setup_logger("RegexExtractor", "extraction.log")
        self.patterns = REGEX_PATTERNS

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

        for node_name, node_config in nodes_config.items():
            node_type_config = NODE_TYPES.get(node_name, {})
            default_extractor = node_type_config.get("default_extractor")

            if default_extractor != ExtractorType.REGEX:
                continue

            extracted = self._extract_by_regex(text, node_name, node_config, node_type_config)
            result.nodes.extend(extracted)

        self.logger.info(f"Extracted {len(result.nodes)} nodes using regex from chunk {chunk_id}")
        return result

    def _extract_by_regex(
        self,
        text: str,
        node_name: str,
        node_config: Dict[str, Any],
        node_type_config: Dict[str, Any]
    ) -> List[ExtractedNode]:
        nodes = []
        attributes = node_config.get("attributes", [])
        cypher_label = node_type_config.get("cypher_label", node_name)
        category = node_type_config.get("category", NodeCategory.S)

        for attr in attributes:
            pattern_key = self._get_pattern_key(attr)
            if pattern_key not in self.patterns:
                continue

            pattern = self.patterns[pattern_key]
            matches = re.finditer(pattern, text)

            for match in matches:
                matched_text = match.group(0).strip()

                node_id = self.id_gen.generate_node_id()
                merge_keys = self._generate_merge_keys(cypher_label, {attr: matched_text})

                node = ExtractedNode(
                    node_id=node_id,
                    node_type=node_name,
                    node_label=node_type_config.get("label", node_name),
                    cypher_label=cypher_label,
                    attributes={attr: matched_text},
                    merge_keys=merge_keys,
                    category=category.value,
                    confidence=1.0,
                    extraction_method="regex"
                )
                nodes.append(node)

        return nodes

    def _get_pattern_key(self, attr: str) -> str:
        if attr in ["chainage"]:
            return "chainage"
        elif attr in ["time"]:
            return "datetime"
        return attr

    def _generate_merge_keys(self, cypher_label: str, attributes: Dict[str, Any]) -> List[str]:
        merge_keys = [f"{cypher_label}"]
        for key, value in attributes.items():
            if value:
                merge_keys.append(f"{key}:{value}")
        return merge_keys
