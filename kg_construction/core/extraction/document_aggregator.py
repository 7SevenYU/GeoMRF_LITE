from typing import Dict, List, Any, Callable
from kg_construction.core.extraction.base_extractor import ExtractedNode, ExtractedRelation, ExtractionResult
from kg_construction.core.extraction.config import DOCUMENT_AGGREGATION_CONFIG
from kg_construction.core.extraction.id_generator import IDGenerator
from kg_construction.utils.logger import setup_logger
from datetime import datetime
import re


class DocumentAggregator:
    def __init__(self):
        self.logger = setup_logger("DocumentAggregator", "extraction.log")
        self.id_gen = IDGenerator()

    def aggregate_by_document(
        self,
        extraction_results: List[ExtractionResult],
        document_type: str
    ) -> List[ExtractionResult]:
        """
        按文档聚合抽取结果

        Args:
            extraction_results: 该文档类型的所有抽取结果
            document_type: 文档类型

        Returns:
            聚合后的抽取结果列表
        """
        agg_config = DOCUMENT_AGGREGATION_CONFIG.get(document_type, {})
        aggregation_level = agg_config.get("aggregation_level", "chunk")

        if aggregation_level == "chunk":
            self.logger.info(f"Document type {document_type} uses chunk-level aggregation, skipping document aggregation")
            return extraction_results

        self.logger.info(f"Aggregating {len(extraction_results)} chunk results for document type {document_type}")

        node_strategies = agg_config.get("node_strategies", {})

        document_groups = self._group_by_source_file(extraction_results)

        aggregated_results = []
        for source_file, results in document_groups.items():
            aggregated = self._aggregate_single_document(results, node_strategies, source_file, document_type)
            if aggregated:
                aggregated_results.append(aggregated)

        self.logger.info(f"Aggregated into {len(aggregated_results)} document-level results")
        return aggregated_results

    def _group_by_source_file(self, extraction_results: List[ExtractionResult]) -> Dict[str, List[ExtractionResult]]:
        """按source_file分组"""
        groups = {}
        for result in extraction_results:
            source_file = result.source_file
            if source_file not in groups:
                groups[source_file] = []
            groups[source_file].append(result)
        return groups

    def _aggregate_single_document(
        self,
        results: List[ExtractionResult],
        node_strategies: Dict[str, str],
        source_file: str,
        document_type: str
    ) -> ExtractionResult:
        """聚合单个文档的所有chunk结果"""
        aggregated = ExtractionResult(
            chunk_id=f"{document_type}_{source_file}_aggregated",
            document_type=document_type,
            source_file=source_file
        )

        all_nodes = []
        for result in results:
            all_nodes.extend(result.nodes)

        nodes_by_type = self._group_nodes_by_type(all_nodes)

        for node_type, strategy in node_strategies.items():
            if node_type not in nodes_by_type:
                continue

            nodes = nodes_by_type[node_type]
            self.logger.info(f"Aggregating {len(nodes)} nodes of type {node_type} using strategy {strategy}")

            if strategy == "merge_descriptions":
                merged_node = self._merge_descriptions(nodes, node_type, source_file)
            elif strategy == "merge_attributes":
                merged_node = self._merge_attributes(nodes, node_type, source_file)
            elif strategy == "select_earliest":
                merged_node = self._select_earliest(nodes, node_type, source_file)
            elif strategy == "select_most_common":
                merged_node = self._select_most_common(nodes, node_type, source_file)
            elif strategy == "create_single":
                merged_node = self._create_single(nodes, node_type, source_file)
            elif strategy == "merge_unique":
                merged_nodes = self._merge_unique(nodes, node_type, source_file)
                aggregated.nodes.extend(merged_nodes)
                continue
            elif strategy == "merge_all":
                merged_nodes = self._merge_all(nodes, node_type, source_file)
                aggregated.nodes.extend(merged_nodes)
                continue
            else:
                self.logger.warning(f"Unknown aggregation strategy: {strategy}")
                continue

            if merged_node:
                aggregated.nodes.append(merged_node)

        relations = self._rebuild_relations(results, aggregated.nodes)
        aggregated.relations.extend(relations)

        self.logger.info(f"Aggregated document {source_file}: {len(aggregated.nodes)} nodes, {len(aggregated.relations)} relations")
        return aggregated

    def _group_nodes_by_type(self, nodes: List[ExtractedNode]) -> Dict[str, List[ExtractedNode]]:
        """按节点类型分组"""
        groups = {}
        for node in nodes:
            node_type = node.node_type
            if node_type not in groups:
                groups[node_type] = []
            groups[node_type].append(node)
        return groups

    def _merge_descriptions(self, nodes: List[ExtractedNode], node_type: str, source_file: str) -> ExtractedNode:
        """合并描述性属性"""
        if not nodes:
            return None

        merged_attributes = {}
        first_node = nodes[0]

        for attr in first_node.attributes.keys():
            values = []
            for node in nodes:
                if attr in node.attributes and node.attributes[attr]:
                    values.append(node.attributes[attr])

            unique_values = list(set(values))
            if unique_values:
                merged_attributes[attr] = "；".join(unique_values)

        merge_key = f"{node_type}_{source_file}"

        merged_node = ExtractedNode(
            node_type=node_type,
            attributes=merged_attributes,
            merge_keys=[merge_key],
            source_chunks=[node.source_chunk for node in nodes]
        )

        return merged_node

    def _merge_attributes(self, nodes: List[ExtractedNode], node_type: str, source_file: str) -> ExtractedNode:
        """
        合并所有属性，保留所有section提取的信息

        特别用于变更信息节点：合并section 1的chainage和section 2的information
        """
        if not nodes:
            return None

        merged_attributes = {}

        # 收集所有节点中的所有属性名
        all_attributes = set()
        for node in nodes:
            all_attributes.update(node.attributes.keys())

        # 合并所有节点的所有属性
        for attr in all_attributes:
            values = []
            for node in nodes:
                if attr in node.attributes and node.attributes[attr]:
                    values.append(node.attributes[attr])

            # 去重，保留唯一值
            unique_values = list(set(values))
            if unique_values:
                # 对于大多数属性，只保留第一个值
                # 但对于变更信息，我们需要保留chainage和information
                if node_type == "变更信息" and attr in ["chainage", "information"]:
                    # 这两个属性都需要保留
                    merged_attributes[attr] = unique_values[0]
                    self.logger.info(f"  Merged {node_type} attribute '{attr}': {unique_values[0]}")
                else:
                    # 其他属性保留第一个
                    merged_attributes[attr] = unique_values[0]

        # 生成merge_key（使用所有属性）
        merge_key = f"{node_type}_{source_file}"

        # 记录source_chunks
        source_chunks = list(set([node.source_chunk for node in nodes if hasattr(node, 'source_chunk')]))

        self.logger.info(f"  Merged {len(nodes)} {node_type} nodes with {len(merged_attributes)} attributes: {list(merged_attributes.keys())}")

        return ExtractedNode(
            node_type=node_type,
            attributes=merged_attributes,
            merge_keys=[merge_key],
            source_chunks=source_chunks
        )

    def _select_earliest(self, nodes: List[ExtractedNode], node_type: str, source_file: str) -> ExtractedNode:
        """选择时间最早的节点"""
        if not nodes:
            return None

        earliest_node = None
        earliest_time = None

        for node in nodes:
            time_attr = node.attributes.get("time", "")
            if time_attr:
                parsed_time = self._parse_time(time_attr)
                if parsed_time:
                    if earliest_time is None or parsed_time < earliest_time:
                        earliest_time = parsed_time
                        earliest_node = node

        if earliest_node:
            merge_key = f"{node_type}_{source_file}"
            earliest_node.merge_keys = [merge_key]
            earliest_node.source_chunks = [node.source_chunk for node in nodes]
            return earliest_node

        if nodes:
            merge_key = f"{node_type}_{source_file}"
            nodes[0].merge_keys = [merge_key]
            nodes[0].source_chunks = [node.source_chunk for node in nodes]
            return nodes[0]

        return None

    def _parse_time(self, time_str: str) -> datetime:
        """解析时间字符串"""
        patterns = [
            r"(\d{4})[-年](\d{1,2})[-月](\d{1,2})",
            r"(\d{4})-(\d{1,2})-(\d{1,2})",
        ]

        for pattern in patterns:
            match = re.search(pattern, time_str)
            if match:
                try:
                    year, month, day = match.groups()
                    return datetime(int(year), int(month), int(day))
                except:
                    continue

        return None

    def _merge_unique(self, nodes: List[ExtractedNode], node_type: str, source_file: str) -> List[ExtractedNode]:
        """合并去重"""
        unique_nodes = {}

        for node in nodes:
            attr_str = str(sorted(node.attributes.items()))
            if attr_str not in unique_nodes:
                unique_nodes[attr_str] = node
            else:
                unique_nodes[attr_str].source_chunks.extend(node.source_chunks)

        result = list(unique_nodes.values())
        for node in result:
            node.merge_keys = [f"{node_type}_{hash(str(sorted(node.attributes.items())))}"]

        return result

    def _create_single(self, nodes: List[ExtractedNode], node_type: str, source_file: str) -> ExtractedNode:
        """
        创建单个节点（用于历史处置案例）

        从多个section的信息合并成一个历史处置案例节点
        """
        if not nodes:
            return None

        merged_attributes = {}
        node_type_config = NODE_TYPES.get(node_type, {})
        cypher_label = node_type_config.get("cypher_label", node_type)
        category = node_type_config.get("category", "S").value

        # 收集所有属性
        all_attributes = set()
        for node in nodes:
            all_attributes.update(node.attributes.keys())

        # 对每个属性，收集所有值
        for attr in all_attributes:
            values = []
            for node in nodes:
                if attr in node.attributes and node.attributes[attr]:
                    values.append(node.attributes[attr])

            if values:
                # 取第一个值（因为是同一个案例的不同section）
                merged_attributes[attr] = values[0]

        # 生成merge_key
        merge_key = f"{cypher_label}_{source_file}"

        # 生成s_id
        from kg_construction.core.extraction.id_generator import IDGenerator
        id_gen = IDGenerator()
        s_id = id_gen.generate_node_id()

        merged_node = ExtractedNode(
            node_id=s_id,
            node_type=node_type,
            node_label=node_type_config.get("label", node_type),
            cypher_label=cypher_label,
            attributes=merged_attributes,
            merge_keys=[merge_key],
            category=category,
            confidence=0.85,
            extraction_method="llm"
        )

        # 重新生成merge_keys，包含s_id
        new_merge_keys = [cypher_label, f"s_id:{s_id}"]
        merged_node.merge_keys = new_merge_keys

        self.logger.info(f"  Created single {node_type} node: {s_id} with {len(merged_attributes)} attributes")

        return merged_node

    def _merge_all(self, nodes: List[ExtractedNode], node_type: str, source_file: str) -> List[ExtractedNode]:
        """保留所有节点"""
        for node in nodes:
            node.merge_keys = [f"{node_type}_{hash(str(sorted(node.attributes.items())))}_{node.source_chunk}"]
        return nodes

    def _select_most_common(self, nodes: List[ExtractedNode], node_type: str, source_file: str) -> ExtractedNode:
        """选择出现最频繁的节点（假设是主要信息）"""
        if not nodes:
            return None

        if len(nodes) == 1:
            merge_key = f"{node_type}_{source_file}"
            nodes[0].merge_keys = [merge_key]
            nodes[0].source_chunks = [nodes[0].source_chunk]
            return nodes[0]

        attr_counts = {}
        for node in nodes:
            for attr, value in node.attributes.items():
                if value:
                    if attr not in attr_counts:
                        attr_counts[attr] = {}
                    if value not in attr_counts[attr]:
                        attr_counts[attr][value] = 0
                    attr_counts[attr][value] += 1

        merged_attributes = {}
        for attr, counts in attr_counts.items():
            most_common_value = max(counts.items(), key=lambda x: x[1])[0]
            merged_attributes[attr] = most_common_value

            max_count = max(counts.values())
            if max_count == 1 and len(counts) > 1:
                self.logger.warning(
                    f"All {node_type} '{attr}' values appear only once, "
                    f"selecting first: {list(counts.keys())[0]}"
                )
            else:
                self.logger.info(
                    f"Selected most common {node_type} '{attr}': {most_common_value} "
                    f"(appears {max_count} times)"
                )

        merge_key = f"{node_type}_{source_file}"
        new_node = ExtractedNode(
            node_id=self.id_gen.generate_node_id(),
            merge_keys=[merge_key],
            node_type=node_type,
            node_label=nodes[0].node_label,
            cypher_label=nodes[0].cypher_label,
            attributes=merged_attributes,
            category=nodes[0].category,
            confidence=0.9,
            extraction_method="llm_aggregated"
        )
        new_node.source_chunks = [node.source_chunk for node in nodes]

        return new_node

    def _rebuild_relations(
        self,
        chunk_results: List[ExtractionResult],
        aggregated_nodes: List[ExtractedNode]
    ) -> List[ExtractedRelation]:
        """重新建立关系"""
        relations = []

        node_type_to_nodes = {}
        for node in aggregated_nodes:
            if node.node_type not in node_type_to_nodes:
                node_type_to_nodes[node.node_type] = []
            node_type_to_nodes[node.node_type].append(node)

        all_relations = []
        for result in chunk_results:
            all_relations.extend(result.relations)

        for rel in all_relations:
            head_type = rel.head_type
            tail_type = rel.tail_type

            if head_type in node_type_to_nodes and tail_type in node_type_to_nodes:
                for head_node in node_type_to_nodes[head_type]:
                    for tail_node in node_type_to_nodes[tail_type]:
                        new_rel = ExtractedRelation(
                            relation_type=rel.relation_type,
                            head=head_node.merge_keys[0] if head_node.merge_keys else None,
                            tail=tail_node.merge_keys[0] if tail_node.merge_keys else None,
                            head_type=head_type,
                            tail_type=tail_type
                        )
                        relations.append(new_rel)

        return relations
