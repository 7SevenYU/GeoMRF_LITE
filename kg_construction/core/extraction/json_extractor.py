import json
from typing import Dict, List, Any
from kg_construction.core.extraction.base_extractor import BaseExtractor, ExtractionResult, ExtractedNode, ExtractedRelation
from kg_construction.core.extraction.config import NODE_TYPES, RELATION_TYPES
from kg_construction.core.extraction.lexicon_extractor import LexiconExtractor
from kg_construction.utils.logger import setup_logger


class JSONExtractor(BaseExtractor):
    def __init__(self, config: Dict[str, Any], id_generator):
        super().__init__(config, id_generator)
        self.logger = setup_logger("JSONExtractor", "extraction.log")
        self.lexicon_extractor = LexiconExtractor(config, id_generator)

    def extract(self, text: str, **kwargs) -> ExtractionResult:
        chunk_id = kwargs.get("chunk_id", "")
        document_type = kwargs.get("document_type", "")
        source_file = kwargs.get("source_file", "")
        metadata = kwargs.get("metadata", {})

        result = ExtractionResult(
            chunk_id=chunk_id,
            document_type=document_type,
            source_file=source_file
        )

        try:
            json_data = json.loads(text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON for chunk {chunk_id}: {e}")
            return result

        doc_config = self.config.get(document_type, {})
        nodes_config = doc_config.get("nodes", {})
        relations_config = doc_config.get("relations", {})

        extracted_nodes, extracted_relations = self._extract_nodes_with_relations(
            json_data, nodes_config, relations_config, chunk_id, document_type, metadata
        )
        result.nodes.extend(extracted_nodes)
        result.relations.extend(extracted_relations)

        self.logger.info(
            f"Extracted {len(result.nodes)} nodes and {len(result.relations)} relations from JSON chunk {chunk_id}"
        )
        return result

    def _extract_nodes_with_relations(
        self,
        json_data: Dict[str, Any],
        nodes_config: Dict[str, Any],
        relations_config: Dict[str, Any],
        chunk_id: str,
        document_type: str,
        metadata: Dict[str, Any] = None
    ) -> tuple[List[ExtractedNode], List[ExtractedRelation]]:
        """
        提取节点并立即建立关系（用于数组元素）

        对于数组节点，每提取一个元素就立即建立该元素的关系
        """
        if metadata is None:
            metadata = {}

        all_nodes = []
        all_relations = []
        parent_chainage = None

        # 第一轮：提取所有非数组节点（全局节点：时间等，探测方法现在是数组节点）
        non_array_nodes = []
        array_node_configs = {}

        for node_name, node_config in nodes_config.items():
            is_array = node_config.get("is_array", False)
            if not is_array:
                # 提取非数组节点
                metadata_field = node_config.get("metadata_field", None)
                node_metadata = metadata.copy()
                if metadata_field == "parent_chainage" and parent_chainage:
                    node_metadata["parent_chainage"] = parent_chainage

                node = self._extract_single_node(
                    json_data, node_name, node_config, [], chunk_id, document_type, node_metadata
                )
                if node:
                    non_array_nodes.append(node)

                # 特殊处理：如果是设计信息节点，提取chainage供子节点使用
                if node_name == "设计信息":
                    composite_fields = node_config.get("composite_fields", {})
                    if "chainage" in composite_fields:
                        fields = composite_fields["chainage"]["fields"]
                        separator = composite_fields["chainage"].get("separator", "～")
                        values = []
                        for field in fields:
                            value = self._get_value_by_path(json_data, field, node_name, field)
                            if value is not None:
                                values.append(str(value))
                        if len(values) == len(fields):
                            parent_chainage = separator.join(values)
            else:
                # 记录数组节点配置
                array_node_configs[node_name] = node_config

        all_nodes.extend(non_array_nodes)

        # 为非数组节点建立关系
        non_array_relations = self._extract_relations(
            json_data, relations_config, non_array_nodes, chunk_id, document_type
        )
        all_relations.extend(non_array_relations)

        # 第二轮：检查是否有数组节点需要提取
        if not array_node_configs:
            return all_nodes, all_relations

        # 找到数组字段名（假设所有数组节点都从同一个字段提取）
        array_field = None
        for node_config in array_node_configs.values():
            array_field = node_config.get("array_field", "")
            if array_field:
                break

        if not array_field:
            return all_nodes, all_relations

        # 遍历数组，为每个元素提取节点并建立关系
        array_items = json_data.get(array_field, [])
        if not isinstance(array_items, list):
            return all_nodes, all_relations

        for item in array_items:
            # 提取这个元素的所有数组节点
            element_nodes = []
            construction_info_node = None

            for node_name, node_config in array_node_configs.items():
                node_metadata = metadata.copy()

                # 提取节点
                node = self._extract_single_node(
                    item, node_name, node_config, [], chunk_id, document_type, node_metadata
                )

                if node:
                    # 对于数组节点，如果有metadata_field，需要手动从metadata获取
                    # 因为_extract_single_node中的metadata_field逻辑是为非数组节点设计的
                    metadata_field = node_config.get("metadata_field", None)
                    if metadata_field and metadata.get(metadata_field):
                        first_attr = node_config.get("attributes", [])[0]
                        if first_attr and first_attr not in node.attributes:
                            node.attributes[first_attr] = metadata[metadata_field]
                            node.merge_keys.append(f"{first_attr}:{metadata[metadata_field]}")

                    # 保存施工信息节点引用，供探测方法使用
                    if node.node_type == "施工信息":
                        construction_info_node = node

                    # 如果是探测方法节点，从施工信息复制chainage属性（仅在节点没有chainage时）
                    if node.node_type == "探测方法" and construction_info_node:
                        chainage_value = construction_info_node.attributes.get("chainage")
                        if chainage_value and "chainage" not in node.attributes:
                            # 只在节点没有 chainage 时才从施工信息复制
                            node.attributes["chainage"] = chainage_value
                            # 更新merge_keys，添加chainage
                            node.merge_keys.append(f"chainage:{chainage_value}")

                    element_nodes.append(node)

            # 添加到总节点列表
            all_nodes.extend(element_nodes)

        # 按组建立关系（每4个节点一组：施工信息、探测方法、探测结论、地质风险等级）
        array_nodes = all_nodes[len(non_array_nodes):]  # 跳过全局节点，获取所有数组节点
        global_nodes_dict = {node.node_type: node for node in non_array_nodes}

        # 每4个节点为一组，对应"结论"数组的每个元素
        for i in range(0, len(array_nodes), 4):
            group_nodes = array_nodes[i:i+4]

            # 找到本组的节点
            construction_info_node = None
            detection_method_node = None
            detection_conclusion_node = None
            risk_grade_node = None

            for node in group_nodes:
                if node.node_type == "施工信息":
                    construction_info_node = node
                elif node.node_type == "探测方法":
                    detection_method_node = node
                elif node.node_type == "探测结论":
                    detection_conclusion_node = node
                elif node.node_type == "地质风险等级":
                    risk_grade_node = node

            # 为本组节点建立关系
            time_node = global_nodes_dict.get("时间")

            if construction_info_node:
                # 施工信息 → 时间（如果有）
                if time_node:
                    all_relations.append(self._create_relation("HAS_SPATIOTEMPORAL", construction_info_node, time_node))

                # 施工信息 → 探测方法（本组的）
                if detection_method_node:
                    all_relations.append(self._create_relation("WAS_SURVEYED_BY", construction_info_node, detection_method_node))

            if detection_method_node:
                # 探测方法 → 探测结论（本组的）
                if detection_conclusion_node:
                    all_relations.append(self._create_relation("INDICATES", detection_method_node, detection_conclusion_node))

                # 探测方法 → 地质风险等级（本组的）
                if risk_grade_node:
                    all_relations.append(self._create_relation("INDICATES", detection_method_node, risk_grade_node))

        return all_nodes, all_relations

    def _extract_nodes(
        self,
        json_data: Dict[str, Any],
        nodes_config: Dict[str, Any],
        chunk_id: str,
        document_type: str,
        metadata: Dict[str, Any] = None
    ) -> List[ExtractedNode]:
        if metadata is None:
            metadata = {}

        nodes = []
        parent_chainage = None

        for node_name, node_config in nodes_config.items():
            json_field_mapping = node_config.get("json_field_mapping", {})
            use_lexicon_for = node_config.get("use_lexicon_for", [])
            is_array = node_config.get("is_array", False)
            metadata_field = node_config.get("metadata_field", None)

            # 如果是设计信息节点，提取chainage供子节点使用
            if node_name == "设计信息":
                composite_fields = node_config.get("composite_fields", {})
                if "chainage" in composite_fields:
                    fields = composite_fields["chainage"]["fields"]
                    separator = composite_fields["chainage"].get("separator", "～")
                    values = []
                    for field in fields:
                        value = self._get_value_by_path(json_data, field, node_name, field)
                        if value is not None:
                            values.append(str(value))
                    if len(values) == len(fields):
                        parent_chainage = separator.join(values)

            # 如果配置了metadata_field且值为parent_chainage，添加到metadata
            node_metadata = metadata.copy()
            if metadata_field == "parent_chainage" and parent_chainage:
                node_metadata["parent_chainage"] = parent_chainage

            if is_array:
                array_field = node_config.get("array_field", "")
                array_items = json_data.get(array_field, [])
                if isinstance(array_items, list):
                    for item in array_items:
                        node = self._extract_single_node(
                            item, node_name, node_config, use_lexicon_for, chunk_id, document_type, node_metadata
                        )
                        if node:
                            nodes.append(node)
            else:
                node = self._extract_single_node(
                    json_data, node_name, node_config, use_lexicon_for, chunk_id, document_type, node_metadata
                )
                if node:
                    nodes.append(node)

        return nodes

    def _extract_single_node(
        self,
        json_data: Dict[str, Any],
        node_name: str,
        node_config: Dict[str, Any],
        use_lexicon_for: List[str],
        chunk_id: str,
        document_type: str,
        metadata: Dict[str, Any] = None
    ) -> ExtractedNode:
        try:
            if metadata is None:
                metadata = {}

            json_field_mapping = node_config.get("json_field_mapping", {})
            composite_fields = node_config.get("composite_fields", {})
            metadata_field = node_config.get("metadata_field", None)
            attributes = {}

            # 如果配置了从metadata读取字段
            if metadata_field:
                metadata_value = metadata.get(metadata_field, "")
                if metadata_value:
                    # 检查是否有json_field_mapping
                    if json_field_mapping:
                        # 为json_field_mapping中值为metadata_field的属性赋值
                        for attr, json_field in json_field_mapping.items():
                            if json_field == metadata_field:
                                attributes[attr] = metadata_value
                    else:
                        # 如果没有json_field_mapping，直接使用attributes中第一个属性名
                        if node_config.get("attributes"):
                            attr_name = node_config["attributes"][0]
                            attributes[attr_name] = metadata_value

            # 从JSON数据中提取字段
            array_to_string = node_config.get("array_to_string", False)
            for attr, json_field in json_field_mapping.items():
                # 如果该属性已从metadata中提取，跳过
                if attr in attributes:
                    continue

                # 特殊处理：__ALL_REMAINING__ 标记，收集所有剩余字段
                if json_field == "__ALL_REMAINING__":
                    remaining_fields = self._collect_remaining_fields(
                        json_data, json_field_mapping, attributes, metadata_field
                    )
                    if remaining_fields:
                        attributes[attr] = remaining_fields
                    continue

                value = self._get_value_by_path(json_data, json_field, node_name, attr)
                if value is not None:
                    # 如果配置了array_to_string且值是数组，转换为字符串
                    if array_to_string and isinstance(value, list):
                        value = json.dumps(value, ensure_ascii=False)
                    attributes[attr] = value

            # 处理组合字段（composite_fields）
            for attr, composite_config in composite_fields.items():
                if attr in attributes:
                    continue  # 如果已经提取过，跳过
                fields = composite_config.get("fields", [])
                separator = composite_config.get("separator", "～")

                # 提取所有字段值
                values = []
                for field in fields:
                    value = self._get_value_by_path(json_data, field, node_name, field)
                    if value is not None:
                        values.append(str(value))

                # 如果所有字段都有值，组合它们
                if len(values) == len(fields):
                    attributes[attr] = separator.join(values)
                    self.logger.debug(f"Composite field {attr} composed: {attributes[attr]}")

            if not attributes:
                return None

            if use_lexicon_for:
                for attr in use_lexicon_for:
                    if attr in attributes:
                        text_value = str(attributes[attr])
                        lexicon_nodes = self._extract_with_lexicon(
                            text_value, node_name, node_config, attr, chunk_id, document_type
                        )
                        if lexicon_nodes:
                            return lexicon_nodes[0]

            node_type_config = NODE_TYPES.get(node_name, {})
            node_id = self.id_gen.generate_node_id()
            cypher_label = node_type_config.get("cypher_label", node_name)
            category = node_type_config.get("category", "S").value

            # 应用值标准化
            normalization = node_type_config.get("normalization", None)
            if normalization == "remove_suffix":
                attributes = self._normalize_attributes(attributes)

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
                extraction_method="json"
            )

            return node

        except Exception as e:
            self.logger.error(f"Error extracting node {node_name} from JSON: {e}")
            return None

    def _extract_with_lexicon(
        self,
        text: str,
        node_name: str,
        node_config: Dict[str, Any],
        attr: str,
        chunk_id: str,
        document_type: str
    ) -> List[ExtractedNode]:
        try:
            result = self.lexicon_extractor.extract(
                text,
                chunk_id=chunk_id,
                document_type=document_type,
                source_file=""
            )
            return result.nodes
        except Exception as e:
            self.logger.error(f"Error extracting with lexicon: {e}")
            return []

    def _extract_relations(
        self,
        json_data: Dict[str, Any],
        relations_config: Dict[str, Any],
        existing_nodes: List[ExtractedNode],
        chunk_id: str,
        document_type: str
    ) -> List[ExtractedRelation]:
        relations = []

        for rel_key, rel_config in relations_config.items():
            json_defined = rel_config.get("json_defined", False)

            if not json_defined:
                continue

            relation_type = rel_config.get("relation", rel_key)
            head_type = rel_config.get("head")
            tail_type = rel_config.get("tail")

            head_nodes = self._find_nodes_by_type(existing_nodes, head_type)
            tail_nodes = self._find_nodes_by_type(existing_nodes, tail_type)

            # Debug logging
            self.logger.info(
                f"[{chunk_id}] Processing relation {relation_type}: "
                f"{head_type} ({len(head_nodes)} nodes) -> {tail_type} ({len(tail_nodes)} nodes)"
            )

            if len(head_nodes) == 1 and len(tail_nodes) == 1:
                relations.append(self._create_relation(relation_type, head_nodes[0], tail_nodes[0]))
            elif len(head_nodes) == 1 and len(tail_nodes) > 1:
                for tail_node in tail_nodes:
                    relations.append(self._create_relation(relation_type, head_nodes[0], tail_node))
            elif len(head_nodes) > 1 and len(tail_nodes) == 1:
                for head_node in head_nodes:
                    relations.append(self._create_relation(relation_type, head_node, tail_nodes[0]))
            elif len(head_nodes) > 1 and len(tail_nodes) > 1:
                min_len = min(len(head_nodes), len(tail_nodes))
                for i in range(min_len):
                    relations.append(self._create_relation(relation_type, head_nodes[i], tail_nodes[i]))

        return relations

    def _create_relation(
        self,
        relation_type: str,
        head_node: ExtractedNode,
        tail_node: ExtractedNode
    ) -> ExtractedRelation:
        relation_id = self.id_gen.generate_relation_id()
        relation_type_config = RELATION_TYPES.get(relation_type, {})

        return ExtractedRelation(
            relation_id=relation_id,
            relation_type=relation_type,
            relation_label=relation_type_config.get("label", relation_type),
            cypher_label=relation_type_config.get("cypher_label", relation_type),
            head_node_id=head_node.node_id,
            tail_node_id=tail_node.node_id,
            head_merge_key=head_node.merge_keys[0] if head_node.merge_keys else "",
            tail_merge_key=tail_node.merge_keys[0] if tail_node.merge_keys else "",
            confidence=1.0,
            extraction_method="json"
        )

    def _find_nodes_by_type(self, nodes: List[ExtractedNode], node_type: str) -> List[ExtractedNode]:
        return [node for node in nodes if node.node_type == node_type]

    def _get_value_by_path(self, data, path, node_name, attr):
        """
        智能路径解析，支持三种方式：
        1. 简单key自动查找
        2. 点号路径明确访问
        3. 索引访问数组元素
        """
        if not path:
            return None

        if '.' not in path and '[' not in path:
            return self._find_key_recursive(data, path, node_name, attr)

        parts = self._parse_path(path)
        value = data

        for part in parts:
            value = self._navigate_to_value(value, part)
            if value is None:
                return None

        # 如果返回值是数组或字典，序列化为JSON字符串
        if isinstance(value, (list, dict)):
            import json
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception as e:
                self.logger.warning(f"Failed to serialize {attr} to JSON: {e}")
                return str(value)

        return value

    def _parse_path(self, path: str):
        """
        解析路径字符串为组件列表
        "风险评估[0].风险类型" -> ["风险评估", "[0]", "风险类型"]
        "设计信息.里程" -> ["设计信息", "里程"]
        """
        parts = []
        current = ""
        bracket_mode = False

        for char in path:
            if char == '[':
                if current:
                    parts.append(current)
                    current = ""
                bracket_mode = True
            elif char == ']':
                parts.append(f"[{current}]")
                current = ""
                bracket_mode = False
            elif char == '.' and not bracket_mode:
                if current:
                    parts.append(current)
                    current = ""
            else:
                current += char

        if current:
            parts.append(current)

        return parts

    def _navigate_to_value(self, data, part):
        """
        根据路径组件导航到值
        支持字典访问和数组索引
        """
        if part.startswith('[') and part.endswith(']'):
            index = int(part[1:-1])
            if isinstance(data, list) and 0 <= index < len(data):
                return data[index]
            return None

        if isinstance(data, dict) and part in data:
            return data[part]

        return None

    def _find_key_recursive(self, data, key, node_name, attr):
        """
        递归在嵌套字典/列表中查找key
        返回第一个找到的值，发现多个同名key时记录警告
        """
        findings = []

        def _search(obj, target_key, current_path=""):
            if isinstance(obj, dict):
                if target_key in obj:
                    findings.append((current_path or target_key, obj[target_key]))
                for k, v in obj.items():
                    _search(v, target_key, current_path + "." + k if current_path else k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        _search(item, target_key, f"{current_path}[{i}]" if current_path else f"[{i}]")

        _search(data, key)

        if len(findings) == 0:
            return None
        elif len(findings) == 1:
            return findings[0][1]
        else:
            self.logger.warning(
                f"Multiple keys '{key}' found for {node_name}.{attr}: "
                f"[{', '.join(f[0] for f in findings)}], using first: {findings[0][0]}"
            )
            return findings[0][1]

    def _collect_remaining_fields(
        self,
        json_data: Dict[str, Any],
        json_field_mapping: Dict[str, str],
        extracted_attrs: Dict[str, Any],
        metadata_field: str = None
    ) -> str:
        """
        收集JSON中所有未被映射的字段，格式化为字符串

        Args:
            json_data: 原始JSON数据
            json_field_mapping: 字段映射配置
            extracted_attrs: 已提取的属性
            metadata_field: metadata字段名

        Returns:
            格式化的字符串，包含所有剩余字段
        """
        # 获取所有已映射的JSON字段名
        mapped_json_fields = set()
        for json_field in json_field_mapping.values():
            if json_field != "__ALL_REMAINING__":
                mapped_json_fields.add(json_field)

        # 收集剩余字段
        remaining_items = []
        for key, value in json_data.items():
            # 跳过已映射的字段、数组字段（结论等）、URL字段
            if key in mapped_json_fields:
                continue
            if isinstance(value, list):
                continue
            if isinstance(value, str) and (value.startswith("http://") or value.startswith("https://")):
                continue
            if value is None or value == "" or value == " ":
                continue

            # 格式化字段
            remaining_items.append(f"{key}:{value}")

        return "；".join(remaining_items)

    def _normalize_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化属性值
        - 去除"级"后缀
        - 统一罗马数字格式（Ⅴ→V, Ⅳ→IV, Ⅲ→III, Ⅱ→II, Ⅰ→I）
        """
        normalized = {}
        roman_mapping = {
            'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV', 'Ⅴ': 'V',
            'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII', 'Ⅸ': 'IX', 'Ⅹ': 'X'
        }

        for key, value in attributes.items():
            if isinstance(value, str):
                # 去除"级"后缀
                if value.endswith('级'):
                    value = value[:-1]

                # 统一罗马数字
                for old, new in roman_mapping.items():
                    value = value.replace(old, new)

                # 去除空格
                value = value.strip()

            normalized[key] = value

        return normalized

    def _generate_merge_keys(self, cypher_label: str, attributes: Dict[str, Any]) -> List[str]:
        merge_keys = [cypher_label]
        for key, value in attributes.items():
            if value:
                merge_keys.append(f"{key}:{value}")
        return merge_keys
