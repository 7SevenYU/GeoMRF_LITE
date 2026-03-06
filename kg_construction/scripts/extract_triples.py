import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# 智能检测项目根目录，支持两种运行方式
script_dir = Path(__file__).parent
if script_dir.name == "scripts":
    # 从项目根目录运行：python -m kg_construction.scripts.extract_triples
    project_root = script_dir.parent.parent
else:
    # 在scripts目录直接运行：python extract_triples.py（调试模式）
    project_root = script_dir.parent.parent

# 确保项目根目录在sys.path中
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from kg_construction.core.extraction.entity_extractor import EntityExtractor
from kg_construction.core.extraction.document_aggregator import DocumentAggregator
from kg_construction.core.extraction.config import DOCUMENT_AGGREGATION_CONFIG
from kg_construction.core.extraction.base_extractor import ExtractionResult
from kg_construction.core.extraction.extraction_tracker import ExtractionTracker
from kg_construction.utils.logger import setup_logger


class TripleExtractor:
    def __init__(self, chunks_dir: str, output_dir: str):
        # 将相对路径转换为基于project_root的绝对路径
        if not Path(chunks_dir).is_absolute():
            chunks_dir = project_root / chunks_dir
        if not Path(output_dir).is_absolute():
            output_dir = project_root / output_dir

        self.chunks_dir = Path(chunks_dir)
        self.output_dir = Path(output_dir)

        self.logger = setup_logger("TripleExtractor", "extraction.log")
        self.entity_extractor = EntityExtractor()
        self.document_aggregator = DocumentAggregator()
        self.extraction_tracker = ExtractionTracker()

        self.results_dir = self.output_dir
        self.graph_data_dir = self.output_dir / "knowledge_graph_data"
        self.graph_data_dir.mkdir(parents=True, exist_ok=True)

        self.global_nodes = {}
        self.global_relations = []

    def process_all_chunks(self):
        self.logger.info(f"开始处理分块数据，源目录: {self.chunks_dir}")

        chunk_files = list(self.chunks_dir.glob("*_chunks.json"))
        self.logger.info(f"找到 {len(chunk_files)} 个分块文件")

        for chunk_file in chunk_files:
            self._process_chunk_file(chunk_file)

        self._build_global_graph_data()
        self._save_global_graph_data()

        self.logger.info("三元组抽取完成")

    def _process_chunk_file(self, chunk_file: Path):
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)

            document_type = chunk_data.get("document_type")
            chunks = chunk_data.get("chunks", [])

            self.logger.info(f"处理 {document_type}: {len(chunks)} 个文本块")

            doc_output_dir = self.results_dir / document_type
            doc_output_dir.mkdir(parents=True, exist_ok=True)

            needs_merge = any(
                chunk.get("metadata", {}).get("segmentation", {}).get("strategy") == "title_based"
                for chunk in chunks
            )

            if needs_merge:
                self.logger.info(f"  检测到title_based分段，使用section合并逻辑")
                self._process_and_merge(chunks, doc_output_dir, document_type)
            else:
                self.logger.info(f"  使用常规chunk处理逻辑")
                self._process_regular_chunks(chunks, doc_output_dir, document_type)

        except Exception as e:
            self.logger.error(f"处理分块文件失败 {chunk_file}: {e}")

    def _process_regular_chunks(self, chunks: List[Dict], output_dir: Path, document_type: str):
        # 按source_file分组
        chunks_by_source = defaultdict(list)
        for chunk in chunks:
            source_file = chunk.get("source_file")
            chunks_by_source[source_file].append(chunk)

        processed_count = 0
        skipped_count = 0

        for source_file, file_chunks in chunks_by_source.items():
            # 检查是否已提取
            if self.extraction_tracker.is_extracted(source_file):
                info = self.extraction_tracker.get_extraction_info(source_file)
                self.logger.info(
                    f"  跳过已提取文件: {Path(source_file).name} "
                    f"({info.get('node_count', 0)} nodes, {info.get('relation_count', 0)} relations)"
                )
                skipped_count += 1
                # 合并已提取的结果
                self._merge_existing_result(source_file, document_type)
                continue

            self.logger.info(f"    处理文件: {Path(source_file).name}")

            # 提取该文件的所有chunks
            chunk_results = []
            for chunk in file_chunks:
                result = self._extract_from_chunk(chunk, output_dir)
                if result:
                    chunk_results.append(result)

            # 聚合
            if chunk_results:
                aggregated_results = self.document_aggregator.aggregate_by_document(
                    chunk_results, document_type
                )

                # 统计节点和关系总数
                total_nodes = sum(len(r.nodes) for r in aggregated_results)
                total_relations = sum(len(r.relations) for r in aggregated_results)

                # 检查是否需要保存聚合结果（仅对document级聚合）
                from kg_construction.core.extraction.config import DOCUMENT_AGGREGATION_CONFIG
                agg_config = DOCUMENT_AGGREGATION_CONFIG.get(document_type, {})
                aggregation_level = agg_config.get("aggregation_level", "chunk")

                if aggregation_level == "document":
                    # 文档级聚合：保存到aggregated子目录
                    self._save_aggregated_results(aggregated_results, output_dir)
                else:
                    # chunk级聚合：chunk文件本身就是结果，不需要额外保存
                    self.logger.info(f"      文档类型 {document_type} 使用chunk级聚合，chunk文件即为最终结果")

                for agg_result in aggregated_results:
                    self._merge_to_global(agg_result)

                # 记录提取状态
                self.extraction_tracker.mark_extracted(
                    source_file=source_file,
                    chunk_count=len(file_chunks),
                    node_count=total_nodes,
                    relation_count=total_relations,
                    extraction_method=aggregation_level,
                    document_type=document_type
                )
                processed_count += 1

        if processed_count > 0 or skipped_count > 0:
            self.logger.info(f"  提取完成: 新处理 {processed_count} 个文件, 跳过 {skipped_count} 个已提取文件")

    def _process_and_merge(self, chunks: List[Dict], output_dir: Path, document_type: str):
        chunks_by_source = defaultdict(list)
        for chunk in chunks:
            source_file = chunk.get("source_file")
            chunks_by_source[source_file].append(chunk)

        processed_count = 0
        skipped_count = 0

        for source_file, section_chunks in chunks_by_source.items():
            # 检查是否已提取
            if self.extraction_tracker.is_extracted(source_file):
                info = self.extraction_tracker.get_extraction_info(source_file)
                self.logger.info(
                    f"  跳过已提取文档: {Path(source_file).name} "
                    f"({info['node_count']} nodes, {info['relation_count']} relations)"
                )
                skipped_count += 1
                # 将已提取的结果也合并到全局图数据中
                self._merge_existing_result(source_file, document_type)
                continue

            self.logger.info(f"    处理文档: {Path(source_file).name}")

            extraction_method = "section_by_section"  # 默认值
            merged_result = None

            # 对于变更纪要，尝试合并所有section文本后一次性提取
            if document_type == "变更纪要":
                try:
                    self.logger.info(f"      尝试使用合并文本提取（一次性提取所有节点）")
                    merged_result = self._extract_with_merged_text(section_chunks, source_file, document_type)
                    if merged_result:
                        extraction_method = "merged_text"
                        self._save_final_result(merged_result, output_dir)
                        self._merge_to_global(merged_result)
                        self.logger.info(f"      合并文本提取成功")
                except Exception as e:
                    from kg_construction.core.extraction.llm_client import TokenLimitError
                    if isinstance(e, TokenLimitError):
                        self.logger.warning(f"      合并文本提取超过token限制，回退到分块提取模式")
                    else:
                        self.logger.error(f"      合并文本提取失败: {e}，回退到分块提取模式")
                    merged_result = None  # 重置状态，确保回退到分section提取

            # 如果合并文本提取失败，使用分section提取方式
            if merged_result is None:
                try:
                    # 回退到分section提取方式
                    section_results = []
                    for chunk in section_chunks:
                        result = self._extract_from_chunk(chunk, output_dir)
                        if result:
                            section_results.append(result)

                    if section_results:
                        merged_result = self._merge_results(section_results, source_file, document_type)
                        if merged_result:
                            extraction_method = "section_by_section"
                            self._save_final_result(merged_result, output_dir)
                            self._merge_to_global(merged_result)
                except Exception as e:
                    self.logger.error(f"      分section提取失败: {e}")
                    merged_result = None  # 确保异常时重置状态

            # 提取成功后记录状态
            if merged_result:
                try:
                    self.extraction_tracker.mark_extracted(
                        source_file=source_file,
                        chunk_count=len(section_chunks),
                        node_count=len(merged_result.nodes),
                        relation_count=len(merged_result.relations),
                        extraction_method=extraction_method,
                        document_type=document_type
                    )
                    processed_count += 1
                except Exception as e:
                    self.logger.error(f"      记录提取状态失败: {e}")

        self.logger.info(f"  提取完成: 新处理 {processed_count} 个文档, 跳过 {skipped_count} 个已提取文档")

    def _extract_with_merged_text(self, section_chunks: List[Dict], source_file: str,
                                  document_type: str) -> ExtractionResult:
        """合并所有section文本后一次性提取所有节点"""
        from kg_construction.core.extraction.llm_client import TokenLimitError

        # 合并所有section的文本
        merged_text = ""
        for i, chunk in enumerate(section_chunks):
            text = chunk.get("text", "")
            section_info = chunk.get("metadata", {}).get("section_info", {})
            section_title = section_info.get("section_title", f"Section {i}")

            # 添加section标题和内容
            merged_text += f"\n【{section_title}】\n{text}\n"

        self.logger.info(f"      合并了 {len(section_chunks)} 个section，总文本长度: {len(merged_text)} 字符")

        # 创建一个合并后的chunk对象
        merged_chunk = {
            "chunk_id": f"{Path(source_file).stem}_merged",
            "text": merged_text.strip(),
            "document_type": document_type,
            "source_file": source_file,
            "metadata": {
                "merged_sections": True,
                "section_count": len(section_chunks),
                "skip_save": True  # 不保存中间结果
            }
        }

        # 使用合并后的文本进行提取
        result = self._extract_from_chunk(merged_chunk, self.output_dir / document_type)

        if result:
            self.logger.info(f"      合并文本提取完成: {len(result.nodes)} 个节点, {len(result.relations)} 个关系")
        else:
            raise Exception("合并文本提取返回空结果")

        return result

    def _merge_results(self, section_results: List[ExtractionResult], source_file: str,
                       document_type: str) -> ExtractionResult:
        self.logger.info(f"      合并 {len(section_results)} 个section结果")

        # 收集节点并记录section信息
        all_nodes = []
        all_relations = []

        # section_results是按顺序的：chunk_000000, chunk_000001, chunk_000002, chunk_000003, chunk_000004
        # 对应section: 0, 1, 2, 3, 4
        for section_idx, result in enumerate(section_results):
            # 为每个节点添加section信息（通过修改node或临时存储）
            # 我们用一个临时字典来记录节点来自哪个section
            for node in result.nodes:
                # 将section索引存储在节点的extraction_method字段中（临时使用）
                # 格式: "llm_section_3"
                if hasattr(node, 'extraction_method') and node.extraction_method:
                    node.extraction_method = f"{node.extraction_method}_section_{section_idx}"
                all_nodes.append(node)
            all_relations.extend(result.relations)

        # 特殊处理：变更纪要的节点去重和合并
        if document_type == "变更纪要":
            all_nodes = self._merge_change_info_nodes(all_nodes)
            all_nodes = self._merge_emergency_response_nodes(all_nodes)
            all_nodes = self._deduplicate_single_value_nodes(all_nodes)

        # 恢复extraction_method字段（去掉section信息）
        for node in all_nodes:
            if hasattr(node, 'extraction_method') and '_section_' in node.extraction_method:
                node.extraction_method = node.extraction_method.split('_section_')[0]

        merged_chunk_id = f"{Path(source_file).stem}_final"

        merged_result = ExtractionResult(
            chunk_id=merged_chunk_id,
            document_type=document_type,
            source_file=source_file,
            nodes=all_nodes,
            relations=all_relations
        )

        self.logger.info(f"      合并完成: {len(all_nodes)} 个节点, {len(all_relations)} 个关系")

        return merged_result

    def _merge_change_info_nodes(self, nodes: List) -> List:
        """合并变更纪要中的变更信息节点属性"""
        from kg_construction.core.extraction.base_extractor import ExtractedNode

        change_info_nodes = [n for n in nodes if n.node_type == "变更信息"]
        other_nodes = [n for n in nodes if n.node_type != "变更信息"]

        if not change_info_nodes:
            return nodes

        # 合并所有变更信息节点的属性
        merged_attributes = {}
        merged_node_id = None
        merged_cypher_label = None
        merged_category = None
        merged_confidence = 0.0
        merged_extraction_method = "llm"

        for node in change_info_nodes:
            for attr, value in node.attributes.items():
                if attr not in merged_attributes:
                    merged_attributes[attr] = value
            if node.confidence > merged_confidence:
                merged_confidence = node.confidence
                merged_node_id = node.node_id
                merged_cypher_label = node.cypher_label
                merged_category = node.category
                # 恢复extraction_method（去掉section信息）
                if hasattr(node, 'extraction_method') and '_section_' in node.extraction_method:
                    merged_extraction_method = node.extraction_method.split('_section_')[0]

        # 创建合并后的变更信息节点
        if merged_attributes:
            merged_node = ExtractedNode(
                node_id=merged_node_id or change_info_nodes[0].node_id,
                node_type="变更信息",
                node_label="变更信息",
                cypher_label=merged_cypher_label or "变更信息",
                attributes=merged_attributes,
                merge_keys=change_info_nodes[0].merge_keys,
                category=merged_category or "S",
                confidence=merged_confidence,
                extraction_method=merged_extraction_method
            )
            # 重新生成merge_keys（现在包含完整的属性）
            from kg_construction.core.extraction.config import NODE_TYPES
            node_type_config = NODE_TYPES.get("变更信息", {})
            merge_key_attrs = node_type_config.get("merge_key_attributes", [])

            new_merge_keys = [merged_node.cypher_label]
            if merge_key_attrs:
                for attr in merge_key_attrs:
                    if attr in merged_attributes and merged_attributes[attr]:
                        new_merge_keys.append(f"{attr}:{merged_attributes[attr]}")
            else:
                for key, value in merged_attributes.items():
                    if value:
                        new_merge_keys.append(f"{key}:{value}")
            merged_node.merge_keys = new_merge_keys

            other_nodes.append(merged_node)
            self.logger.info(
                f"      合并变更信息节点: {len(change_info_nodes)} -> 1, 属性: {list(merged_attributes.keys())}")

        return other_nodes

    def _merge_emergency_response_nodes(self, nodes: List) -> List:
        """合并变更纪要中的紧急响应措施节点"""
        from kg_construction.core.extraction.base_extractor import ExtractedNode

        emergency_nodes = [n for n in nodes if n.node_type == "紧急响应措施"]
        other_nodes = [n for n in nodes if n.node_type != "紧急响应措施"]

        if len(emergency_nodes) <= 1:
            return nodes

        # 使用merge_descriptions策略：将多个节点的属性值用"；"连接
        merged_attributes = {}
        first_node = emergency_nodes[0]

        # 收集所有节点的所有属性名
        all_attributes = set()
        for node in emergency_nodes:
            all_attributes.update(node.attributes.keys())

        # 对每个属性，收集所有值并根据类型进行合并
        for attr in all_attributes:
            values = []
            for node in emergency_nodes:
                if attr in node.attributes and node.attributes[attr]:
                    values.append(node.attributes[attr])

            unique_values = list(set(values))
            if unique_values:
                # 特殊处理数组类型属性（如keywords）
                if attr == "keywords" and any(isinstance(v, list) for v in unique_values):
                    # 合并数组：展平并去重
                    flattened_keywords = []
                    for value in unique_values:
                        if isinstance(value, list):
                            flattened_keywords.extend(value)
                        else:
                            flattened_keywords.append(value)
                    merged_attributes[attr] = list(set(flattened_keywords))
                elif isinstance(unique_values[0], list):
                    # 其他数组类型属性，跳过合并
                    self.logger.warning(f"      跳过数组类型属性: {attr}")
                    continue
                else:
                    # 字符串类型属性用"；"连接
                    merged_attributes[attr] = "；".join(unique_values)

        # 创建合并后的节点
        merged_node = ExtractedNode(
            node_id=first_node.node_id,
            node_type="紧急响应措施",
            node_label=first_node.node_label,
            cypher_label=first_node.cypher_label,
            attributes=merged_attributes,
            merge_keys=first_node.merge_keys,
            category=first_node.category,
            confidence=first_node.confidence,
            extraction_method=first_node.extraction_method
        )

        # 重新生成merge_keys（现在包含完整的合并属性）
        new_merge_keys = [merged_node.cypher_label]
        for key, value in merged_attributes.items():
            if value:
                new_merge_keys.append(f"{key}:{value}")
        merged_node.merge_keys = new_merge_keys

        other_nodes.append(merged_node)
        self.logger.info(f"      合并紧急响应措施节点: {len(emergency_nodes)} -> 1")

        return other_nodes

    def _deduplicate_single_value_nodes(self, nodes: List) -> List:
        """去重单值节点类型，优先保留section 4的节点"""
        # 单值节点类型：只有一个属性的节点，值必须唯一
        single_value_node_types = {
            "时间", "围岩等级", "风险类型", "探测方法",
            "预警等级", "地质风险等级", "风险评估"
        }

        # 按node_type分组
        nodes_by_type = {}
        for node in nodes:
            node_type = node.node_type
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)

        result_nodes = []

        for node_type, node_list in nodes_by_type.items():
            if node_type in single_value_node_types and len(node_list) > 1:
                # 单值节点类型有多个实例，优先保留section 4的
                section_4_node = None
                highest_section_idx = -1

                for node in node_list:
                    # 从extraction_method中提取section索引
                    section_idx = -1
                    if hasattr(node, 'extraction_method') and '_section_' in node.extraction_method:
                        try:
                            section_idx = int(node.extraction_method.split('_section_')[1])
                        except (ValueError, IndexError):
                            pass

                    # 选择section索引最大的（section 4的索引最大）
                    if section_idx > highest_section_idx:
                        highest_section_idx = section_idx
                        section_4_node = node

                if section_4_node:
                    result_nodes.append(section_4_node)
                    self.logger.info(
                        f"      去重 {node_type}: {len(node_list)} -> 1 (保留section {highest_section_idx}的)")
            else:
                # 非单值节点类型或只有1个实例，保留所有
                result_nodes.extend(node_list)

        return result_nodes

    def _save_final_result(self, result: ExtractionResult, output_dir: Path):
        source_filename = Path(result.source_file).stem
        output_file = output_dir / f"{source_filename}.json"

        output_data = {
            "chunk_id": result.chunk_id,
            "document_type": result.document_type,
            "source_file": result.source_file,
            "merged": True,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "cypher_label": node.cypher_label,
                    "attributes": node.attributes,
                    "merge_keys": node.merge_keys,
                    "category": node.category,
                    "confidence": node.confidence,
                    "extraction_method": node.extraction_method
                }
                for node in result.nodes
            ],
            "relations": [
                {
                    "relation_id": rel.relation_id,
                    "relation_type": rel.relation_type,
                    "cypher_label": rel.cypher_label,
                    "head_node_id": rel.head_node_id,
                    "tail_node_id": rel.tail_node_id,
                    "confidence": rel.confidence,
                    "extraction_method": rel.extraction_method
                }
                for rel in result.relations
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"      保存最终结果: {output_file}")

    def _extract_from_chunk(self, chunk: Dict[str, Any], output_dir: Path):
        try:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            document_type = chunk.get("document_type")
            source_file = chunk.get("source_file")
            metadata = chunk.get("metadata", {})

            result = self.entity_extractor.extract(
                text=text,
                document_type=document_type,
                chunk_id=chunk_id,
                source_file=source_file,
                metadata=metadata
            )

            skip_save = metadata.get("skip_save", False)
            if not skip_save:
                self._save_chunk_result(result, output_dir)
            else:
                self.logger.info(f"      跳过保存中间结果: {chunk_id}")

            return result

        except Exception as e:
            self.logger.error(f"抽取失败 {chunk.get('chunk_id')}: {e}")
            return None

    def _save_chunk_result(self, result, output_dir: Path):
        chunk_filename = f"{result.chunk_id}.json"
        output_file = output_dir / chunk_filename

        output_data = {
            "chunk_id": result.chunk_id,
            "document_type": result.document_type,
            "source_file": result.source_file,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "cypher_label": node.cypher_label,
                    "attributes": node.attributes,
                    "merge_keys": node.merge_keys,
                    "category": node.category,
                    "confidence": node.confidence,
                    "extraction_method": node.extraction_method
                }
                for node in result.nodes
            ],
            "relations": [
                {
                    "relation_id": rel.relation_id,
                    "relation_type": rel.relation_type,
                    "cypher_label": rel.cypher_label,
                    "head_node_id": rel.head_node_id,
                    "tail_node_id": rel.tail_node_id,
                    "confidence": rel.confidence,
                    "extraction_method": rel.extraction_method
                }
                for rel in result.relations
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def _save_aggregated_results(self, aggregated_results: List[ExtractionResult], output_dir: Path):
        """保存聚合后的文档级结果"""
        agg_output_dir = output_dir / "aggregated"
        agg_output_dir.mkdir(parents=True, exist_ok=True)

        for result in aggregated_results:
            source_filename = Path(result.source_file).stem
            output_file = agg_output_dir / f"{source_filename}_aggregated.json"

            output_data = {
                "chunk_id": result.chunk_id,
                "document_type": result.document_type,
                "source_file": result.source_file,
                "aggregated": True,
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "node_type": node.node_type,
                        "cypher_label": node.cypher_label,
                        "attributes": node.attributes,
                        "merge_keys": node.merge_keys,
                        "category": node.category,
                        "confidence": node.confidence,
                        "extraction_method": node.extraction_method,
                        "source_chunks": node.source_chunks
                    }
                    for node in result.nodes
                ],
                "relations": [
                    {
                        "relation_id": rel.relation_id,
                        "relation_type": rel.relation_type,
                        "cypher_label": rel.cypher_label,
                        "head_node_id": rel.head_node_id,
                        "tail_node_id": rel.tail_node_id,
                        "confidence": rel.confidence,
                        "extraction_method": rel.extraction_method
                    }
                    for rel in result.relations
                ]
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"保存聚合结果到 {agg_output_dir}")

    def _merge_existing_result(self, source_file: str, document_type: str):
        """将已提取的结果合并到全局图数据中"""
        source_filename = Path(source_file).stem
        result_file = self.results_dir / document_type / f"{source_filename}.json"

        if not result_file.exists():
            self.logger.warning(f"      已提取文档的结果文件不存在: {result_file}")
            return

        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)

            # 构造ExtractionResult对象
            result = ExtractionResult(
                chunk_id=result_data.get("chunk_id", source_filename),
                document_type=result_data.get("document_type", document_type),
                source_file=result_data.get("source_file", source_file)
            )

            # 重建节点对象
            from kg_construction.core.extraction.base_extractor import ExtractedNode
            for node_data in result_data.get("nodes", []):
                node = ExtractedNode(
                    node_id=node_data["node_id"],
                    node_type=node_data["node_type"],
                    node_label=node_data.get("node_label", node_data["node_type"]),
                    cypher_label=node_data["cypher_label"],
                    attributes=node_data["attributes"],
                    merge_keys=node_data["merge_keys"],
                    category=node_data["category"],
                    confidence=node_data["confidence"],
                    extraction_method=node_data.get("extraction_method", "llm")
                )
                result.nodes.append(node)

            # 重建关系对象
            from kg_construction.core.extraction.base_extractor import ExtractedRelation
            for rel_data in result_data.get("relations", []):
                rel = ExtractedRelation(
                    relation_id=rel_data["relation_id"],
                    relation_type=rel_data["relation_type"],
                    relation_label=rel_data.get("relation_label", rel_data["relation_type"]),
                    cypher_label=rel_data["cypher_label"],
                    head_node_id=rel_data["head_node_id"],
                    tail_node_id=rel_data["tail_node_id"],
                    confidence=rel_data["confidence"],
                    extraction_method=rel_data.get("extraction_method", "llm")
                )
                result.relations.append(rel)

            # 合并到全局图数据
            self._merge_to_global(result)
            self.logger.info(
                f"      已合并已提取文档的结果: {len(result.nodes)} nodes, {len(result.relations)} relations")

        except Exception as e:
            self.logger.error(f"      合并已提取结果失败: {e}")

    def _merge_to_global(self, result):
        for node in result.nodes:
            merge_key_str = "|".join(node.merge_keys)

            if merge_key_str not in self.global_nodes:
                self.global_nodes[merge_key_str] = {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "cypher_label": node.cypher_label,
                    "attributes": node.attributes,
                    "merge_keys": node.merge_keys,
                    "category": node.category,
                    "confidence": node.confidence,
                    "extraction_method": node.extraction_method,
                    "source_chunks": [],
                    "all_node_ids": []
                }

            global_node = self.global_nodes[merge_key_str]
            if result.chunk_id not in global_node["source_chunks"]:
                global_node["source_chunks"].append(result.chunk_id)
            if node.node_id not in global_node["all_node_ids"]:
                global_node["all_node_ids"].append(node.node_id)

            if node.confidence > global_node["confidence"]:
                global_node["confidence"] = node.confidence

        for rel in result.relations:
            self.global_relations.append({
                "relation_id": rel.relation_id,
                "relation_type": rel.relation_type,
                "relation_label": rel.relation_label,
                "cypher_label": rel.cypher_label,
                "head_node_id": rel.head_node_id,
                "tail_node_id": rel.tail_node_id,
                "head_merge_key": rel.head_merge_key,
                "tail_merge_key": rel.tail_merge_key,
                "confidence": rel.confidence,
                "extraction_method": rel.extraction_method,
                "source_chunk": result.chunk_id
            })

    def _build_global_graph_data(self):
        """构建全局图数据"""
        self.logger.info("构建全局图数据...")

        graph_data = {
            "nodes": [],
            "relations": []
        }

        for merge_key, node_data in self.global_nodes.items():
            graph_data["nodes"].append({
                "node_id": node_data["node_id"],
                "node_type": node_data["node_type"],
                "cypher_label": node_data["cypher_label"],
                "attributes": node_data["attributes"],
                "merge_keys": node_data["merge_keys"],
                "category": node_data["category"],
                "confidence": node_data["confidence"],
                "extraction_method": node_data["extraction_method"],
                "source_chunks": node_data["source_chunks"],
                "all_node_ids": node_data["all_node_ids"]
            })

        graph_data["relations"] = self.global_relations

        self.logger.info(
            f"全局图数据构建完成: {len(graph_data['nodes'])} 个节点, {len(graph_data['relations'])} 个关系")
        return graph_data

    def _save_global_graph_data(self):
        """保存全局图数据"""
        self.logger.info("保存全局图数据...")

        graph_data = self._build_global_graph_data()

        output_file = self.graph_data_dir / "global_graph.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"全局图数据已保存到: {output_file}")

        return graph_data


if __name__ == "__main__":
    extractor = TripleExtractor("kg_construction/data/processed/chunks", "kg_construction/data/processed/extraction_results")
    extractor.process_all_chunks()
