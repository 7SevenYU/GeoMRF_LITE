"""
Cypher查询生成器：根据推理链路配置生成Neo4j查询
"""

from typing import List
from kg_construction.core.graph_inference.inference_config import RelationChain, LinkStep


class CypherQueryBuilder:
    """根据推理链路生成Cypher查询"""

    @staticmethod
    def build_multi_hop_query(chain: RelationChain) -> str:
        """
        为多跳推理链路生成Cypher查询

        Args:
            chain: 关系链路配置

        Returns:
            Cypher查询字符串
        """
        if len(chain.path) < 2:
            raise ValueError(f"链路 {chain.relation_type} 路径长度小于2，不是多跳链路")

        source_label = chain.source_node_type
        target_label = chain.target_node_type
        max_paths = chain.max_paths

        # 构建路径模式和WHERE子句
        path_pattern, where_clause = CypherQueryBuilder._build_path_elements(chain.path)

        # 生成查询
        query = f"""
MATCH (source:{source_label})
WHERE source.node_id IS NOT NULL
CALL {{
    WITH source
    MATCH path = {path_pattern}
    {where_clause}
    RETURN last(nodes(path)) as target_node
    LIMIT {max_paths}
}}
RETURN
    source.node_id as head_node_id,
    target_node.node_id as tail_node_id,
    '{chain.relation_type}' as relation_type,
    '{chain.cypher_label}' as cypher_label,
    {chain.path[-1].confidence} as confidence
"""
        return query.strip()

    @staticmethod
    def _build_path_elements(steps: List[LinkStep]) -> tuple:
        """
        构建路径模式和WHERE子句

        Args:
            steps: 链路步骤列表

        Returns:
            (path_pattern, where_clause) 元组
        """
        pattern_parts = []
        where_conditions = []

        # 起始节点
        pattern_parts.append(f"(source)")

        for i, step in enumerate(steps):
            if step.type == "explicit_relation":
                # 显式关系：添加关系模式
                rel_pattern = f"-[:{step.relation_type}]->"
                target_node = f"({step.to_node.replace(' ', '_')}_{i}:{step.to_node})"
                pattern_parts.append(rel_pattern)
                pattern_parts.append(target_node)
            elif step.type == "id_match":
                # ID匹配：也使用关系模式，但在WHERE中添加属性相等条件
                # 使用通用关系模式，实际关系类型在WHERE中通过属性匹配来限制
                source_node = f"{step.from_node.replace(' ', '_')}_{i-1}" if i > 0 else "source"
                target_node = f"({step.to_node.replace(' ', '_')}_{i}:{step.to_node})"
                rel_pattern = f"->"  # 使用通用关系
                pattern_parts.append(rel_pattern)
                pattern_parts.append(target_node)
            else:
                # 其他类型暂不支持
                target_node = f"({step.to_node.replace(' ', '_')}_{i}:{step.to_node})"
                rel_pattern = f"->"
                pattern_parts.append(rel_pattern)
                pattern_parts.append(target_node)

            # 添加WHERE条件
            condition = CypherQueryBuilder._build_step_condition(step, i)
            if condition:
                where_conditions.append(condition)

        path_pattern = "".join(pattern_parts)
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

        return path_pattern, where_clause

    @staticmethod
    def _build_step_condition(step: LinkStep, step_index: int) -> str:
        """
        为单个步骤构建WHERE条件

        Args:
            step: 链路步骤
            step_index: 步骤索引

        Returns:
            WHERE条件字符串
        """
        target_var = f"{step.to_node.replace(' ', '_')}_{step_index}"

        if step.type == "id_match":
            source_var = f"{step.from_node.replace(' ', '_')}_{step_index-1}" if step_index > 0 else "source"
            return f"{source_var}.{step.from_attribute} = {target_var}.{step.to_attribute}"

        elif step.type == "explicit_relation":
            # 显式关系不需要额外WHERE条件
            return ""

        elif step.type == "attribute_match":
            source_var = f"{step.from_node.replace(' ', '_')}_{step_index-1}" if step_index > 0 else "source"
            return f"{source_var}.{step.from_attribute} = {target_var}.{step.to_attribute}"

        else:
            # 其他步骤类型暂不支持
            return ""
