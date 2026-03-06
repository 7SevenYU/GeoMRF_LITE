from py2neo import Graph, Node, Relationship
from typing import Dict, List, Optional, Any
import logging
import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from kg_construction.core.extraction.config import NODE_ATTRIBUTE_SCHEMAS


def _get_node_merge_key(node_label: str, attributes: Dict[str, Any]) -> str:
    """
    根据节点类型获取用于merge的业务属性key
    如果attributes中有该属性，则使用该属性作为merge key
    否则使用第一个属性
    """
    # 定义每种节点类型应该使用的merge属性
    merge_key_mapping = {
        "围岩等级": "grade",
        "风险类型": "riskType",
        "预警等级": "warningGrade",
        "地质风险等级": "geologicalRiskGrade",
        "探测方法": "detectionMethod",
        "探测结论": "detectionConclusion",
        "设计信息": "chainage",  # 设计信息使用chainage作为唯一标识
        "施工信息": "chainage",
        "变更信息": "chainage",
        "风险评估": "chainage",  # 风险评估使用chainage区分不同记录
        "紧急响应措施": "emergencyResponseGuidelines",
        "施工规范": "constructionSpecifications",
        "历史处置案例": "caseDescription",
        "时间": "time",
        "里程": "chainage"
    }

    # 获取该节点类型应该使用的merge key
    preferred_key = merge_key_mapping.get(node_label)

    if preferred_key and preferred_key in attributes:
        return preferred_key

    # 如果没有预定义的key或该key不存在，使用第一个属性
    if attributes:
        return list(attributes.keys())[0]

    # 如果没有属性，返回node_id（保底方案）
    return "node_id"


def _serialize_value(value: Any) -> Any:
    """
    将非原始类型的值序列化为JSON字符串
    Neo4j属性只接受原始类型或其数组
    """
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return value


def _normalize_node_attributes(node_label: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据节点类型标准化属性值格式
    - 去除"级"后缀
    - 统一罗马数字格式（Ⅴ→V, Ⅳ→IV, Ⅲ→III, Ⅱ→II, Ⅰ→I）
    - 去除多余空格
    - 对于地质风险等级单例匹配，排除chainage属性
    """
    normalized = {}
    roman_mapping = {
        'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV', 'Ⅴ': 'V',
        'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII', 'Ⅸ': 'IX', 'Ⅹ': 'X'
    }

    for key, value in attributes.items():
        if isinstance(value, str):
            # 去除空格
            value = value.strip()

            # 围岩等级特殊处理
            if node_label == "围岩等级" and key == "grade":
                # 去除"级"后缀
                if value.endswith('级'):
                    value = value[:-1]
                # 统一罗马数字
                for old, new in roman_mapping.items():
                    value = value.replace(old, new)

            # 预警等级、地质风险等级等类似处理
            elif key == "geologicalRiskGrade":
                # 映射 green->Low, yellow->Middle, orange->High, red->Critical
                grade_mapping = {
                    'green': 'Low',
                    'yellow': 'Middle',
                    'orange': 'High',
                    'red': 'Critical'
                }
                value = grade_mapping.get(value, value)
            elif key == "warningGrade":
                if value.endswith('级'):
                    value = value[:-1]

        # 地质风险等级单例匹配：排除chainage属性
        if node_label == "地质风险等级" and key == "chainage":
            continue

        normalized[key] = value

    return normalized


def _standardize_node_attributes(node_label: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化节点属性，确保同一类型节点有相同的属性集合

    对于定义在NODE_ATTRIBUTE_SCHEMAS中的节点类型：
    - 确保包含所有定义的属性key
    - 为缺失的属性填充None

    对于未定义的节点类型：
    - 返回原始属性

    Args:
        node_label: 节点类型标签
        attributes: 原始属性字典

    Returns:
        标准化后的属性字典
    """
    if node_label not in NODE_ATTRIBUTE_SCHEMAS:
        # 未定义schema的节点类型，直接返回原属性
        return attributes

    schema_attrs = NODE_ATTRIBUTE_SCHEMAS[node_label]
    standardized = {}

    # 首先添加所有schema中定义的属性，缺失的设为None
    for attr_key in schema_attrs:
        standardized[attr_key] = attributes.get(attr_key)

    # 然后添加schema中未定义但存在于attributes中的属性（保留额外信息）
    for attr_key, attr_value in attributes.items():
        if attr_key not in standardized:
            standardized[attr_key] = attr_value

    return standardized


def _sanitize_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理属性字典，将所有复杂类型序列化为字符串
    """
    sanitized = {}
    for key, value in attributes.items():
        sanitized[key] = _serialize_value(value)
    return sanitized


class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.graph = None
        self.logger = logging.getLogger(__name__)

    def connect(self):
        try:
            self.graph = Graph(self.uri, auth=(self.username, self.password), name=self.database)
            self.graph.run("RETURN 1")
            self.logger.info(f"成功连接到Neo4j数据库: {self.uri}")
        except Exception as e:
            self.logger.error(f"连接Neo4j数据库失败: {e}")
            raise

    def close(self):
        self.graph = None
        self.logger.info("Neo4j连接已关闭")

    def create_node(self, node_label: str, node_id: str, attributes: Dict[str, Any], merge_keys: List[str]) -> bool:
        try:
            # 单例节点类型：不创建节点，返回True（跳过）
            singleton_types = ["围岩等级", "风险类型", "预警等级", "地质风险等级"]
            if node_label in singleton_types:
                return True  # 跳过单例节点的创建

            # 步骤1: 标准化属性（确保同一类型节点有相同的属性集合）
            standardized_attrs = _standardize_node_attributes(node_label, attributes)
            # 步骤2: 标准化属性格式（如去除"级"后缀、统一罗马数字等）
            normalized_attrs = _normalize_node_attributes(node_label, standardized_attrs)
            # 步骤3: 序列化复杂类型
            sanitized_attrs = _sanitize_attributes(normalized_attrs)

            # 使用 node_id 创建（所有节点都用 node_id 作为 merge key）
            node_attrs = {"node_id": node_id}
            node_attrs.update(sanitized_attrs)
            node = Node(node_label, **node_attrs)
            self.graph.merge(node, node_label, "node_id")
            return True
        except Exception as e:
            self.logger.error(f"创建节点失败: {e}")
            return False

    def create_relation(
        self,
        relation_label: str,
        source_node_id: str,
        target_node_id: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        try:
            query = """
            MATCH (source {node_id: $source_node_id})
            MATCH (target {node_id: $target_node_id})
            MERGE (source)-[r:%s]->(target)
            ON CREATE SET r += $attributes
            ON MATCH SET r += $attributes
            RETURN r
            """ % relation_label

            self.graph.run(
                query,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                attributes=attributes or {}
            )
            return True
        except Exception as e:
            self.logger.error(f"创建关系失败: {e}")
            return False

    def create_nodes_batch(self, nodes: List[Dict[str, Any]]) -> int:
        success_count = 0
        for node in nodes:
            node_label = node["cypher_label"]
            node_id = node["node_id"]
            attributes = node.get("attributes", {})
            merge_keys = node.get("merge_keys", [])

            if not merge_keys:
                self.logger.warning(f"节点 {node_id} 缺少merge_keys，跳过")
                continue

            try:
                # 单例节点类型：不创建节点，直接跳过
                singleton_types = ["围岩等级", "风险类型", "预警等级", "地质风险等级"]
                if node_label in singleton_types:
                    success_count += 1
                    continue

                # 先标准化属性格式
                normalized_attrs = _normalize_node_attributes(node_label, attributes)
                # 再序列化复杂类型
                sanitized_attrs = _sanitize_attributes(normalized_attrs)

                # 使用 node_id 创建（所有节点都用 node_id 作为 merge key）
                node_attrs = {"node_id": node_id}
                node_attrs.update(sanitized_attrs)
                node_obj = Node(node_label, **node_attrs)
                self.graph.merge(node_obj, node_label, "node_id")
                success_count += 1
            except Exception as e:
                self.logger.error(f"批量创建节点失败 {node_id}: {e}")

        return success_count

    def create_relations_batch(self, relations: List[Dict[str, Any]], node_info_map: Dict[str, Dict] = None) -> int:
        success_count = 0
        for relation in relations:
            relation_label = relation["cypher_label"]
            source_node_id = relation["head_node_id"]
            target_node_id = relation["tail_node_id"]
            attributes = {
                "confidence": relation.get("confidence"),
                "extraction_method": relation.get("extraction_method")
            }

            # 获取节点信息
            source_node = node_info_map.get(source_node_id) if node_info_map else None
            target_node = node_info_map.get(target_node_id) if node_info_map else None

            # 检查是否涉及单例节点类型
            source_is_singleton = source_node and source_node.get("cypher_label") in ["围岩等级", "风险类型", "预警等级", "地质风险等级"]
            target_is_singleton = target_node and target_node.get("cypher_label") in ["围岩等级", "风险类型", "预警等级", "地质风险等级"]

            try:
                if source_is_singleton or target_is_singleton:
                    # 涉及单例节点，使用属性匹配创建关系
                    self._create_relation_with_singleton(
                        relation_label, source_node_id, target_node_id,
                        source_node, target_node,
                        source_is_singleton, target_is_singleton, attributes
                    )
                else:
                    # 普通节点，使用 node_id 匹配
                    query = """
                    MATCH (source {node_id: $source_node_id})
                    MATCH (target {node_id: $target_node_id})
                    MERGE (source)-[r:%s]->(target)
                    ON CREATE SET r += $attributes
                    ON MATCH SET r += $attributes
                    RETURN r
                    """ % relation_label

                    self.graph.run(
                        query,
                        source_node_id=source_node_id,
                        target_node_id=target_node_id,
                        attributes=attributes
                    )
                success_count += 1
            except Exception as e:
                self.logger.error(f"批量创建关系失败 {relation.get('relation_id')}: {e}")

        return success_count

    def _create_relation_with_singleton(
        self,
        relation_label: str,
        source_node_id: str,
        target_node_id: str,
        source_node: Dict,
        target_node: Dict,
        source_is_singleton: bool,
        target_is_singleton: bool,
        attributes: Dict[str, Any]
    ):
        """创建涉及单例节点的关系
        通过属性查找单例节点并创建关系
        """
        source_match = self._get_match_clause("source", source_node_id, source_node, source_is_singleton)
        target_match = self._get_match_clause("target", target_node_id, target_node, target_is_singleton)

        query = f"""
        {source_match}
        {target_match}
        MERGE (source)-[r:{relation_label}]->(target)
        ON CREATE SET r += $attributes
        ON MATCH SET r += $attributes
        RETURN r
        """

        self.graph.run(query, attributes=attributes)

    def _get_match_clause(self, var_name: str, node_id: str, node_info: Dict, is_singleton: bool) -> str:
        """根据节点信息生成MATCH子句
        如果是单例节点，通过属性查找；否则通过node_id查找
        """
        if not is_singleton or not node_info:
            return f"MATCH ({var_name} {{node_id: '{node_id}'}})"

        # 单例节点：通过属性查找
        node_label = node_info.get("cypher_label")
        node_attrs = node_info.get("attributes", {})

        # 标准化属性值（去除"级"后缀，统一罗马数字等）
        normalized_attrs = _normalize_node_attributes(node_label, node_attrs)

        # 构建属性匹配条件
        attr_conditions = []
        for key, value in normalized_attrs.items():
            if isinstance(value, str):
                attr_conditions.append(f"{key}: '{value}'")
            else:
                attr_conditions.append(f"{key}: {value}")

        attrs_str = ", ".join(attr_conditions)
        return f"MATCH ({var_name}:{node_label} {{{attrs_str}}})"

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        try:
            result = self.graph.run(query, parameters or {})
            return [record for record in result]
        except Exception as e:
            self.logger.error(f"执行查询失败: {e}")
            raise

    def clear_database(self):
        try:
            self.graph.delete_all()
            self.logger.warning("数据库已清空")
        except Exception as e:
            self.logger.error(f"清空数据库失败: {e}")
            raise

    def get_database_info(self) -> Dict[str, int]:
        try:
            node_count = self.graph.run("MATCH (n) RETURN count(n) as count").data()[0]["count"]
            relation_count = self.graph.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]["count"]
            return {"node_count": node_count, "relation_count": relation_count}
        except Exception as e:
            self.logger.error(f"获取数据库信息失败: {e}")
            return {"node_count": 0, "relation_count": 0}
