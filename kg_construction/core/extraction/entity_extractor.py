from typing import Dict, List, Any
from kg_construction.core.extraction.base_extractor import BaseExtractor, ExtractionResult, ExtractedNode
from kg_construction.core.extraction.config import DOCUMENT_EXTRACTION_CONFIG, ExtractorType, SemanticExtractorType
from kg_construction.core.extraction.regex_extractor import RegexExtractor
from kg_construction.core.extraction.lexicon_extractor import LexiconExtractor
from kg_construction.core.extraction.llm_extractor import LLMExtractor
from kg_construction.core.extraction.json_extractor import JSONExtractor
from kg_construction.core.extraction.id_generator import IDGenerator
from kg_construction.utils.prompt_loader import PromptLoader
from kg_construction.utils.logger import setup_logger


class EntityExtractor:
    def __init__(self):
        self.logger = setup_logger("EntityExtractor", "extraction.log")
        self.config = DOCUMENT_EXTRACTION_CONFIG
        self.id_gen = IDGenerator()
        self.prompt_loader = PromptLoader()

        self.regex_extractor = RegexExtractor(self.config, self.id_gen)
        self.lexicon_extractor = LexiconExtractor(self.config, self.id_gen)
        self.llm_extractor = LLMExtractor(self.config, self.id_gen)
        self.json_extractor = JSONExtractor(self.config, self.id_gen)

    def extract(
        self,
        text: str,
        document_type: str,
        chunk_id: str = "",
        source_file: str = "",
        metadata: Dict[str, Any] = None
    ) -> ExtractionResult:
        # 对于设计信息，根据source_file后缀选择配置
        config_key = document_type
        if document_type == "设计信息":
            if source_file.endswith(".json"):
                config_key = "设计信息(JSON)"
            elif source_file.endswith(".pdf"):
                config_key = "设计信息(PDF)"
        # 对于超前地质预报，根据metadata中的detection_method选择配置
        elif document_type == "超前地质预报":
            detection_method = metadata.get("detection_method", "")
            if detection_method:
                config_key = f"超前地质预报_{detection_method}"
            else:
                self.logger.warning(f"超前地质预报缺少detection_method metadata，使用默认配置")
                config_key = "超前地质预报_水平声波剖面"  # 默认配置

        if config_key not in self.config:
            self.logger.error(f"Unknown document type: {config_key}")
            return ExtractionResult(
                chunk_id=chunk_id,
                document_type=document_type,
                source_file=source_file
            )

        doc_config = self.config[config_key]
        source_format = doc_config.get("source_format", "pdf")

        result = ExtractionResult(
            chunk_id=chunk_id,
            document_type=document_type,
            source_file=source_file
        )

        metadata = metadata or {}
        segmentation = metadata.get("segmentation", {})
        strategy = segmentation.get("strategy", "")

        if strategy == "title_based":
            return self._extract_from_section(
                text, document_type, chunk_id, source_file, segmentation
            )

        if source_format == "json":
            json_result = self.json_extractor.extract(
                text,
                chunk_id=chunk_id,
                document_type=config_key,
                source_file=source_file,
                metadata=metadata
            )
            result.nodes.extend(json_result.nodes)
            result.relations.extend(json_result.relations)
        else:
            regex_result = self.regex_extractor.extract(
                text,
                chunk_id=chunk_id,
                document_type=config_key,
                source_file=source_file
            )
            result.nodes.extend(regex_result.nodes)

            semantic_extractor_type = doc_config.get("semantic_extractor")

            if semantic_extractor_type == SemanticExtractorType.LEXICON:
                lexicon_result = self.lexicon_extractor.extract(
                    text,
                    chunk_id=chunk_id,
                    document_type=config_key,
                    source_file=source_file
                )
                result.nodes.extend(lexicon_result.nodes)

            elif semantic_extractor_type == SemanticExtractorType.LLM:
                llm_result = self.llm_extractor.extract(
                    text,
                    chunk_id=chunk_id,
                    document_type=config_key,
                    source_file=source_file
                )
                result.nodes.extend(llm_result.nodes)
                result.relations.extend(llm_result.relations)

        self._deduplicate_nodes(result)
        self._deduplicate_relations(result)

        self.logger.info(
            f"Total extraction from chunk {chunk_id}: "
            f"{len(result.nodes)} nodes, {len(result.relations)} relations"
        )

        return result

    def _extract_from_section(
        self,
        text: str,
        document_type: str,
        chunk_id: str,
        source_file: str,
        segmentation: Dict[str, Any]
    ) -> ExtractionResult:
        result = ExtractionResult(
            chunk_id=chunk_id,
            document_type=document_type,
            source_file=source_file
        )

        namespace = segmentation.get("namespace")
        section_id = segmentation.get("section_id")

        section_config = self.prompt_loader.get_section_config(document_type, section_id)
        if not section_config:
            self.logger.warning(f"No config for section {section_id} in document type {document_type}")
            return result

        extraction_method = section_config.get("extraction_method", "none")

        if extraction_method == "regex":
            regex_result = self.regex_extractor.extract(
                text,
                chunk_id=chunk_id,
                document_type=document_type,
                source_file=source_file
            )
            result.nodes.extend(regex_result.nodes)
            if section_id == "0":
                self.logger.info(f"Section 0: calling _extract_time_by_regex")
                try:
                    time_nodes = self._extract_time_by_regex(text, chunk_id, source_file)
                    self.logger.info(f"Section 0: _extract_time_by_regex returned {len(time_nodes)} nodes")
                    result.nodes.extend(time_nodes)
                except Exception as e:
                    self.logger.error(f"Section 0: _extract_time_by_regex failed: {e}", exc_info=True)
            elif section_id == "1":
                self.logger.info(f"Section 1: calling _extract_chainage_by_regex")
                try:
                    chainage_nodes = self._extract_chainage_by_regex(text, chunk_id, source_file)
                    self.logger.info(f"Section 1: _extract_chainage_by_regex returned {len(chainage_nodes)} nodes")
                    result.nodes.extend(chainage_nodes)
                except Exception as e:
                    self.logger.error(f"Section 1: _extract_chainage_by_regex failed: {e}", exc_info=True)


        elif extraction_method == "llm":
            prompt = self.prompt_loader.get_section_prompt(document_type, section_id, text)
            if prompt:
                llm_result = self.llm_extractor.extract_with_prompt(
                    prompt,
                    chunk_id=chunk_id,
                    document_type=document_type,
                    source_file=source_file
                )
                result.nodes.extend(llm_result.nodes)
                result.relations.extend(llm_result.relations)

        self._deduplicate_nodes(result)
        self._deduplicate_relations(result)

        self.logger.info(
            f"Section {section_id} extraction: "
            f"{len(result.nodes)} nodes, {len(result.relations)} relations"
        )

        return result

    def batch_extract(
        self,
        chunks: List[Dict[str, str]],
        document_type: str
    ) -> List[ExtractionResult]:
        results = []

        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "")
            source_file = chunk.get("source_file", "")

            result = self.extract(
                text=text,
                document_type=document_type,
                chunk_id=chunk_id,
                source_file=source_file
            )
            results.append(result)

        self.logger.info(f"Batch extraction completed for {len(chunks)} chunks")
        return results

    def extract_from_json_file(
        self,
        json_data: List[Dict[str, Any]],
        document_type: str,
        source_file: str = ""
    ) -> List[ExtractionResult]:
        results = []

        for i, record in enumerate(json_data):
            chunk_id = f"{document_type}_{i}"
            text = str(record)

            result = self.extract(
                text=text,
                document_type=document_type,
                chunk_id=chunk_id,
                source_file=source_file
            )
            results.append(result)

        self.logger.info(f"Extracted from {len(json_data)} JSON records")
        return results

    def _deduplicate_nodes(self, result: ExtractionResult):
        seen_merge_keys = {}
        unique_nodes = []

        for node in result.nodes:
            merge_key_str = "|".join(node.merge_keys)
            if merge_key_str not in seen_merge_keys:
                seen_merge_keys[merge_key_str] = node
                unique_nodes.append(node)
            else:
                existing = seen_merge_keys[merge_key_str]
                if node.confidence > existing.confidence:
                    unique_nodes.remove(existing)
                    unique_nodes.append(node)
                    seen_merge_keys[merge_key_str] = node

        result.nodes = unique_nodes

    def _deduplicate_relations(self, result: ExtractionResult):
        seen_relations = {}
        unique_relations = []

        for relation in result.relations:
            # 使用node_id而不是merge_key进行去重，避免合法的多个关系被错误去重
            rel_key = (
                f"{relation.head_node_id}|"
                f"{relation.relation_type}|"
                f"{relation.tail_node_id}"
            )
            if rel_key not in seen_relations:
                seen_relations[rel_key] = relation
                unique_relations.append(relation)
            else:
                existing = seen_relations[rel_key]
                if relation.confidence > existing.confidence:
                    unique_relations.remove(existing)
                    unique_relations.append(relation)
                    seen_relations[rel_key] = relation

        result.relations = unique_relations

    def _extract_time_by_regex(self, text: str, chunk_id: str, source_file: str) -> List[ExtractedNode]:
        """使用正则表达式提取时间节点（用于段0）"""
        import re
        from kg_construction.core.extraction.config import NODE_TYPES, NodeCategory
        
        nodes = []
        self.logger.info(f"[_extract_time_by_regex] text length: {len(text)}, preview: {text[:100] if text else 'empty'}")
        time_pattern = r'\d{4}[-年]\d{1,2}[-月]\d{1,2}[日号]?'
        
        matches = list(re.finditer(time_pattern, text))
        self.logger.info(f"[_extract_time_by_regex] pattern: '{time_pattern}', matches: {len(matches)}")
        for match in matches:
            matched_text = match.group(0).strip()
            self.logger.info(f"[_extract_time_by_regex] matched: '{matched_text}' at position {match.start()}-{match.end()}")
            
            node_id = self.id_gen.generate_node_id()
            node_type_config = NODE_TYPES.get("时间", {})
            cypher_label = node_type_config.get("cypher_label", "时间")
            
            merge_keys = [cypher_label, f"time:{matched_text}"]
            
            node = ExtractedNode(
                node_id=node_id,
                node_type="时间",
                node_label=node_type_config.get("label", "TIME"),
                cypher_label=cypher_label,
                attributes={"time": matched_text},
                merge_keys=merge_keys,
                category=NodeCategory.S.value,
                confidence=1.0,
                extraction_method="regex"
            )
            nodes.append(node)
            self.logger.info(f"[_extract_time_by_regex] created time node: {node.node_id}")
            break  # 只提取第一个时间
        
        return nodes

    def _extract_chainage_by_regex(self, text: str, chunk_id: str, source_file: str) -> List[ExtractedNode]:
        """使用正则表达式提取里程范围节点（用于段1）"""
        import re
        from kg_construction.core.extraction.config import NODE_TYPES, NodeCategory
        
        nodes = []
        self.logger.info(f"[_extract_chainage_by_regex] text length: {len(text)}, preview: {text[:100] if text else 'empty'}")
        chainage_pattern = r"[A-Z]*K?\d+\+\d+(?:\.\d+)?\s*[～~-~]\s*[A-Z]*K?\d+\+\d+(?:\.\d+)?"
        
        matches = list(re.finditer(chainage_pattern, text))
        self.logger.info(f"[_extract_chainage_by_regex] pattern: '{chainage_pattern}', matches: {len(matches)}")
        for match in matches:
            matched_text = match.group(0).strip()
            self.logger.info(f"[_extract_chainage_by_regex] matched: '{matched_text}' at position {match.start()}-{match.end()}")
            
            node_id = self.id_gen.generate_node_id()
            node_type_config = NODE_TYPES.get("变更信息", {})
            cypher_label = node_type_config.get("cypher_label", "变更信息")
            
            merge_keys = [cypher_label, f"chainage:{matched_text}"]
            
            node = ExtractedNode(
                node_id=node_id,
                node_type="变更信息",
                node_label=node_type_config.get("label", "CHANGE_INFORMATION"),
                cypher_label=cypher_label,
                attributes={"chainage": matched_text},
                merge_keys=merge_keys,
                category=NodeCategory.S.value,
                confidence=1.0,
                extraction_method="regex"
            )
            nodes.append(node)
            self.logger.info(f"[_extract_chainage_by_regex] created chainage node: {node.node_id}")
            break  # 只提取第一个里程范围
        
        return nodes
