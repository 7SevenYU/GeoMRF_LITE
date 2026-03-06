import re
import sys
from pathlib import Path
from typing import Dict, List, Any
from kg_construction.core.extraction.base_extractor import BaseExtractor, ExtractionResult, ExtractedNode
from kg_construction.core.extraction.config import LEXICON_CONFIG, NODE_TYPES, ExtractorType, NodeCategory
from kg_construction.utils.logger import setup_logger

# 导入项目内部的字典提取模块
try:
    from kg_construction.core.extraction.data.lexicons.by_lexicons import (
        GeoConditionExtractor,
        RootLexicon,
        AttrLexicon,
        ExtractConfig
    )
    LEXICONS_AVAILABLE = True
except ImportError:
    GeoConditionExtractor = None
    RootLexicon = None
    AttrLexicon = None
    ExtractConfig = None
    LEXICONS_AVAILABLE = False


class LexiconExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any], id_generator):
        super().__init__(config, id_generator)
        self.logger = setup_logger("LexiconExtractor", "extraction.log")
        self.lexicon_config = LEXICON_CONFIG
        self._init_geo_condition_extractor()

    def _init_geo_condition_extractor(self):
        geo_config = self.lexicon_config.get("geo_condition", {})
        root_csv = geo_config.get("root_csv")
        attr_csv = geo_config.get("attr_csv")

        if root_csv and attr_csv and Path(root_csv).exists() and Path(attr_csv).exists():
            root_lex = RootLexicon(root_csv)
            attr_lex = AttrLexicon(attr_csv)
            cfg = ExtractConfig()
            self.geo_extractor = GeoConditionExtractor(root_lex, attr_lex, cfg)
            self.logger.info("GeoConditionExtractor initialized successfully")
        else:
            self.geo_extractor = None
            self.logger.warning("GeoConditionExtractor not initialized due to missing lexicon files")

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

            if default_extractor != ExtractorType.SEMANTIC:
                continue

            use_lexicon_for = node_config.get("use_lexicon_for", [])
            if not use_lexicon_for:
                continue

            extracted = self._extract_by_lexicon(
                text, node_name, node_config, node_type_config, use_lexicon_for
            )
            result.nodes.extend(extracted)

        self.logger.info(f"Extracted {len(result.nodes)} nodes using lexicon from chunk {chunk_id}")
        return result

    def _extract_by_lexicon(
        self,
        text: str,
        node_name: str,
        node_config: Dict[str, Any],
        node_type_config: Dict[str, Any],
        use_lexicon_for: List[str]
    ) -> List[ExtractedNode]:
        nodes = []
        cypher_label = node_type_config.get("cypher_label", node_name)
        category = node_type_config.get("category", NodeCategory.S).value

        for attr in use_lexicon_for:
            if attr == "geologicalElements":
                extracted = self._extract_geo_conditions(text, node_name, node_type_config)
                nodes.extend(extracted)
            elif attr in ["riskDescription", "detectionConclusion"]:
                extracted = self._extract_geo_conditions(text, node_name, node_type_config)
                nodes.extend(extracted)
            else:
                keywords = self._get_keywords_for_attr(attr)
                if keywords:
                    extracted = self._extract_by_keywords(
                        text, node_name, node_type_config, attr, keywords
                    )
                    nodes.extend(extracted)

        return nodes

    def _extract_geo_conditions(
        self,
        text: str,
        node_name: str,
        node_type_config: Dict[str, Any]
    ) -> List[ExtractedNode]:
        nodes = []

        if not self.geo_extractor:
            return nodes

        try:
            extraction_result = self.geo_extractor.extract(text, block_id="")

            for root in extraction_result.get("roots", []):
                canonical = root.get("canonical", "")
                category = root.get("category", "")

                if not canonical:
                    continue

                node_id = self.id_gen.generate_node_id()
                cypher_label = node_type_config.get("cypher_label", node_name)
                category_value = node_type_config.get("category", NodeCategory.G).value

                attributes = {
                    "geologicalElements": canonical,
                    "geoCategory": category
                }

                merge_keys = self._generate_merge_keys(cypher_label, attributes)

                node = ExtractedNode(
                    node_id=node_id,
                    node_type=node_name,
                    node_label=node_type_config.get("label", node_name),
                    cypher_label=cypher_label,
                    attributes=attributes,
                    merge_keys=merge_keys,
                    category=category_value,
                    confidence=0.9,
                    extraction_method="lexicon_geo"
                )
                nodes.append(node)

        except Exception as e:
            self.logger.error(f"Error extracting geo conditions: {e}")

        return nodes

    def _extract_by_keywords(
        self,
        text: str,
        node_name: str,
        node_type_config: Dict[str, Any],
        attr: str,
        keywords: List[str]
    ) -> List[ExtractedNode]:
        nodes = []

        for keyword in keywords:
            if keyword in text:
                node_id = self.id_gen.generate_node_id()
                cypher_label = node_type_config.get("cypher_label", node_name)
                category = node_type_config.get("category", NodeCategory.C).value

                attributes = {attr: keyword}
                merge_keys = self._generate_merge_keys(cypher_label, attributes)

                node = ExtractedNode(
                    node_id=node_id,
                    node_type=node_name,
                    node_label=node_type_config.get("label", node_name),
                    cypher_label=cypher_label,
                    attributes=attributes,
                    merge_keys=merge_keys,
                    category=category,
                    confidence=1.0,
                    extraction_method="lexicon_keyword"
                )
                nodes.append(node)

        return nodes

    def _get_keywords_for_attr(self, attr: str) -> List[str]:
        attr_to_lexicon = {
            "riskType": "risk_type",
            "surroundingRockGrade": "rock_grade",
            "geologicalRiskGrade": "risk_level",
            "warningGrade": "risk_level",
            "riskAssessment": "risk_level"
        }

        lexicon_key = attr_to_lexicon.get(attr)
        if lexicon_key:
            config = self.lexicon_config.get(lexicon_key, {})
            return config.get("keywords", [])

        return []

    def _generate_merge_keys(self, cypher_label: str, attributes: Dict[str, Any]) -> List[str]:
        merge_keys = [cypher_label]
        for key, value in attributes.items():
            if value:
                merge_keys.append(f"{key}:{value}")
        return merge_keys
