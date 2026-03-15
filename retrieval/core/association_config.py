"""
关联检索配置文件

声明式定义所有关联查询，支持：
- 直接属性查询
- 单跳关系查询
- 多跳路径查询
- 里程匹配查询

每个查询独立执行，具有容错能力
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型"""
    DIRECT_PROPERTY = "direct_property"  # 直接属性查询
    SINGLE_HOP = "single_hop"  # 单跳关系查询
    MULTI_HOP = "multi_hop"  # 多跳路径查询
    MILEAGE_MATCH = "mileage_match"  # 里程匹配查询


class MileageSource(Enum):
    """里程来源"""
    FROM_QUERY = "from_query"  # 来自用户查询
    FROM_NODE = "from_node"  # 来自节点属性


@dataclass
class LinkStep:
    """路径步骤"""
    relation_type: str
    direction: str = "OUT"  # OUT, IN, BOTH
    target_node_type: str = ""
    optional: bool = False  # 是否为可选关系（使用OPTIONAL MATCH）


@dataclass
class ReturnField:
    """返回字段配置"""
    field_name: str  # Neo4j中的属性名
    alias: str  # 返回结果中的别名
    required: bool = False  # 是否必需
    fallback_value: Any = None  # 失败时的回退值
    source: str = ""  # 字段来源节点类型（如"历史处置案例"）
    source_step: int = -1  # 字段来自路径的第几步（-1表示最后一步）


@dataclass
class AssociationQuery:
    """关联查询配置"""
    query_name: str  # 查询名称（唯一标识）
    query_type: QueryType  # 查询类型
    source_node_type: str  # 源节点类型

    # 查询路径配置
    query_path: List[LinkStep] = field(default_factory=list)  # 关系路径
    return_fields: List[ReturnField] = field(default_factory=list)  # 返回字段

    # 里程匹配配置（仅用于MILEAGE_MATCH类型）
    requires_mileage_match: bool = False  # 是否需要里程匹配
    mileage_source: Optional[MileageSource] = None  # 里程来源
    mileage_field: str = ""  # 里程字段名

    # 执行配置
    enabled: bool = True  # 是否启用
    required: bool = False  # 是否必需（必需失败则整个查询失败）
    description: str = ""  # 查询描述


# ============================================================================
# 基于方案ID（Neo4j内部ID）的关联查询配置
# ============================================================================

PLAN_BASED_QUERIES: List[AssociationQuery] = [
    # 1. 直接属性查询
    AssociationQuery(
        query_name="plan_properties",
        query_type=QueryType.DIRECT_PROPERTY,
        source_node_type="紧急响应措施",
        return_fields=[
            ReturnField("node_id", "node_id", required=True),
            ReturnField("applicableConditions", "applicableConditions", required=False),
            ReturnField("emergencyResponseGuidelines", "emergencyResponseGuidelines", required=False),
        ],
        enabled=True,
        required=False,
        description="方案直接属性"
    ),

    # 2. 响应的风险类型（单跳）
    AssociationQuery(
        query_name="risk_types",
        query_type=QueryType.SINGLE_HOP,
        source_node_type="紧急响应措施",
        query_path=[
            LinkStep("RESPONDS_TO", "OUT", "风险类型")
        ],
        return_fields=[
            ReturnField("riskType", "riskType", required=False),
        ],
        enabled=True,
        required=False,
        description="响应的风险类型"
    ),

    # 3. 历史处置案例（多跳，含围岩等级）
    AssociationQuery(
        query_name="historical_cases",
        query_type=QueryType.MULTI_HOP,
        source_node_type="紧急响应措施",
        query_path=[
            LinkStep("RESPONDS_TO", "OUT", "历史处置案例"),
            LinkStep("HAS_SURROUNDING_ROCK_GRADE", "OUT", "围岩等级", optional=True)
        ],
        return_fields=[
            ReturnField("s_id", "s_id", required=False, source="历史处置案例", source_step=0),
            ReturnField("warningDate", "warningDate", required=False, source="历史处置案例", source_step=0),
            ReturnField("riskDescription", "riskDescription", required=False, source="历史处置案例", source_step=0),
            ReturnField("chainage", "chainage", required=False, source="历史处置案例", source_step=0),
            ReturnField("grade", "grade", required=False, source="grade", source_step=1),
        ],
        enabled=True,
        required=False,
        description="历史处置案例（含围岩等级）"
    ),

    # 4. 预警等级（多跳：方案→案例→预警等级）
    AssociationQuery(
        query_name="warning_grade",
        query_type=QueryType.MULTI_HOP,
        source_node_type="紧急响应措施",
        query_path=[
            LinkStep("RESPONDS_TO", "OUT", "历史处置案例"),
            LinkStep("HAS_WARNING_GRADE", "OUT", "预警等级")
        ],
        return_fields=[
            ReturnField("warningGrade", "warningGrade", required=False),
        ],
        enabled=True,
        required=False,
        description="warningGrade"
    ),

    # 5. 施工信息（多跳：方案→案例→施工信息）
    AssociationQuery(
        query_name="construction_info",
        query_type=QueryType.MULTI_HOP,
        source_node_type="紧急响应措施",
        query_path=[
            LinkStep("RESPONDS_TO", "OUT", "历史处置案例"),
            LinkStep("OCCURS_AT", "OUT", "施工信息")
        ],
        return_fields=[
            ReturnField("chainage", "chainage", required=False),
            ReturnField("information", "information", required=False),
        ],
        enabled=True,
        required=False,
        description="information"
    ),

    # 6. 探测方法（多跳：方案→案例→施工信息→探测方法）
    AssociationQuery(
        query_name="detection_methods",
        query_type=QueryType.MULTI_HOP,
        source_node_type="紧急响应措施",
        query_path=[
            LinkStep("RESPONDS_TO", "OUT", "历史处置案例"),
            LinkStep("OCCURS_AT", "OUT", "施工信息"),
            LinkStep("WAS_SURVEYED_BY", "OUT", "探测方法")
        ],
        return_fields=[
            ReturnField("detectionMethod", "detectionMethod", required=False),
            ReturnField("chainage", "chainage", required=False),
        ],
        enabled=True,
        required=False,
        description="detectionMethod"
    ),

    # 7. 探测结论（多跳：方案→案例→施工信息→探测方法→探测结论）
    AssociationQuery(
        query_name="detection_conclusions",
        query_type=QueryType.MULTI_HOP,
        source_node_type="紧急响应措施",
        query_path=[
            LinkStep("RESPONDS_TO", "OUT", "历史处置案例"),
            LinkStep("OCCURS_AT", "OUT", "施工信息"),
            LinkStep("WAS_SURVEYED_BY", "OUT", "探测方法"),
            LinkStep("INDICATES", "OUT", "探测结论")
        ],
        return_fields=[
            ReturnField("detectionConclusion", "detectionConclusion", required=False),
            ReturnField("geologicalElements", "geologicalElements", required=False),
            ReturnField("后续建议", "后续建议", required=False),
        ],
        enabled=True,
        required=False,
        description="detectionConclusion"
    ),

    # 8. 地质风险等级（多跳：方案→案例→施工信息→探测方法→地质风险等级）
    AssociationQuery(
        query_name="geological_risk_levels",
        query_type=QueryType.MULTI_HOP,
        source_node_type="紧急响应措施",
        query_path=[
            LinkStep("RESPONDS_TO", "OUT", "历史处置案例"),
            LinkStep("OCCURS_AT", "OUT", "施工信息"),
            LinkStep("WAS_SURVEYED_BY", "OUT", "探测方法"),
            LinkStep("INDICATES", "OUT", "地质风险等级")
        ],
        return_fields=[
            ReturnField("geologicalRiskGrade", "geologicalRiskGrade", required=False),
        ],
        enabled=True,
        required=False,
        description="geologicalRiskGrade"
    ),

    # 9. 围岩等级（多跳：方案→案例→施工信息→探测方法→围岩等级）- 已禁用，使用historical_cases中的围岩等级
    AssociationQuery(
        query_name="rock_grades",
        query_type=QueryType.MULTI_HOP,
        source_node_type="紧急响应措施",
        query_path=[
            LinkStep("RESPONDS_TO", "OUT", "历史处置案例"),
            LinkStep("OCCURS_AT", "OUT", "施工信息"),
            LinkStep("WAS_SURVEYED_BY", "OUT", "探测方法"),
            LinkStep("INDICATES", "OUT", "围岩等级")
        ],
        return_fields=[
            ReturnField("grade", "grade", required=False),
        ],
        enabled=False,  # 禁用：探测方法没有指向围岩等级的INDICATES关系
        required=False,
        description="围岩等级（已禁用）"
    ),

    # 10. 时间（多跳：方案→案例→施工信息→时间）
    AssociationQuery(
        query_name="time_info",
        query_type=QueryType.MULTI_HOP,
        source_node_type="紧急响应措施",
        query_path=[
            LinkStep("RESPONDS_TO", "OUT", "历史处置案例"),
            LinkStep("OCCURS_AT", "OUT", "施工信息"),
            LinkStep("HAS_SPATIOTEMPORAL", "OUT", "时间")
        ],
        return_fields=[
            ReturnField("time", "time", required=False),
        ],
        enabled=True,
        required=False,
        description="时间信息"
    ),

    # 11. 设计信息（多跳：方案→案例→施工信息→设计信息）
    AssociationQuery(
        query_name="design_info",
        query_type=QueryType.MULTI_HOP,
        source_node_type="紧急响应措施",
        query_path=[
            LinkStep("RESPONDS_TO", "OUT", "历史处置案例"),
            LinkStep("OCCURS_AT", "OUT", "施工信息"),
            LinkStep("IS_ASSOCIATED_WITH", "OUT", "设计信息")
        ],
        return_fields=[
            ReturnField("chainage", "chainage", required=False),
            ReturnField("information", "designInformation", required=False),
            ReturnField("length", "length", required=False),
            ReturnField("grade", "class", required=False),
        ],
        enabled=True,
        required=False,
        description="designInformation"
    ),
]


# ============================================================================
# 基于里程的关联查询配置
# ============================================================================

MILEAGE_BASED_QUERIES: List[AssociationQuery] = [
    # 1. 施工信息（里程匹配）
    AssociationQuery(
        query_name="construction_info",
        query_type=QueryType.MILEAGE_MATCH,
        source_node_type="施工信息",
        mileage_source=MileageSource.FROM_QUERY,
        mileage_field="chainage",
        return_fields=[
            ReturnField("chainage", "chainage", required=False),
            ReturnField("information", "information", required=False),
        ],
        enabled=True,
        required=False,
        description="施工信息（里程匹配）"
    ),

    # 2. 设计信息（里程匹配）
    AssociationQuery(
        query_name="design_info",
        query_type=QueryType.MILEAGE_MATCH,
        source_node_type="设计信息",
        mileage_source=MileageSource.FROM_QUERY,
        mileage_field="chainage",
        return_fields=[
            ReturnField("chainage", "chainage", required=False),
            ReturnField("information", "designInformation", required=False),
            ReturnField("length", "length", required=False),
            ReturnField("grade", "class", required=False),
        ],
        enabled=True,
        required=False,
        description="设计信息（里程匹配）"
    ),

    # 3. 探测方法（里程匹配）
    AssociationQuery(
        query_name="detection_methods",
        query_type=QueryType.MILEAGE_MATCH,
        source_node_type="探测方法",
        mileage_source=MileageSource.FROM_QUERY,
        mileage_field="chainage",
        return_fields=[
            ReturnField("detectionMethod", "detectionMethod", required=False),
            ReturnField("chainage", "chainage", required=False),
        ],
        enabled=True,
        required=False,
        description="探测方法（里程匹配）"
    ),

    # 4. 探测结论（多跳：探测方法→探测结论，里程匹配）
    AssociationQuery(
        query_name="detection_conclusions",
        query_type=QueryType.MULTI_HOP,
        source_node_type="探测方法",
        mileage_source=MileageSource.FROM_QUERY,
        mileage_field="chainage",
        query_path=[
            LinkStep("INDICATES", "OUT", "探测结论")
        ],
        return_fields=[
            ReturnField("detectionConclusion", "detectionConclusion", required=False),
            ReturnField("geologicalElements", "geologicalElements", required=False),
            ReturnField("后续建议", "后续建议", required=False),
        ],
        enabled=True,
        required=False,
        description="探测结论（里程匹配）"
    ),

    # 5. 地质风险等级（多跳：探测方法→地质风险等级，里程匹配）
    AssociationQuery(
        query_name="geological_risk_levels",
        query_type=QueryType.MULTI_HOP,
        source_node_type="探测方法",
        mileage_source=MileageSource.FROM_QUERY,
        mileage_field="chainage",
        query_path=[
            LinkStep("INDICATES", "OUT", "地质风险等级")
        ],
        return_fields=[
            ReturnField("geologicalRiskGrade", "geologicalRiskGrade", required=False),
        ],
        enabled=True,
        required=False,
        description="地质风险等级（里程匹配）"
    ),

    # 6. 围岩等级（多跳：探测方法→围岩等级，里程匹配）- 已禁用，探测方法没有指向围岩等级的INDICATES关系
    AssociationQuery(
        query_name="rock_grades",
        query_type=QueryType.MULTI_HOP,
        source_node_type="探测方法",
        mileage_source=MileageSource.FROM_QUERY,
        mileage_field="chainage",
        query_path=[
            LinkStep("INDICATES", "OUT", "grade")
        ],
        return_fields=[
            ReturnField("grade", "grade", required=False),
        ],
        enabled=False,  # 禁用：探测方法没有指向围岩等级的INDICATES关系
        required=False,
        description="围岩等级（里程匹配，已禁用）"
    ),

    # 7. 历史处置案例（里程匹配）
    AssociationQuery(
        query_name="historical_cases",
        query_type=QueryType.MILEAGE_MATCH,
        source_node_type="历史处置案例",
        mileage_source=MileageSource.FROM_QUERY,
        mileage_field="chainage",
        return_fields=[
            ReturnField("s_id", "s_id", required=False),
            ReturnField("warningDate", "warningDate", required=False),
            ReturnField("riskDescription", "riskDescription", required=False),
            ReturnField("chainage", "chainage", required=False),
        ],
        enabled=True,
        required=False,
        description="历史处置案例（里程匹配）"
    ),

    # 8. 预警等级（多跳：历史处置案例→预警等级，里程匹配）
    AssociationQuery(
        query_name="warning_grades",
        query_type=QueryType.MULTI_HOP,
        source_node_type="历史处置案例",
        mileage_source=MileageSource.FROM_QUERY,
        mileage_field="chainage",
        query_path=[
            LinkStep("HAS_WARNING_GRADE", "OUT", "预警等级")
        ],
        return_fields=[
            ReturnField("warningGrade", "warningGrade", required=False),
        ],
        enabled=True,
        required=False,
        description="预警等级（里程匹配）"
    ),

    # 9. 风险评估（多跳：设计信息→风险评估，里程匹配）
    AssociationQuery(
        query_name="risk_assessments",
        query_type=QueryType.MULTI_HOP,
        source_node_type="设计信息",
        mileage_source=MileageSource.FROM_QUERY,
        mileage_field="chainage",
        query_path=[
            LinkStep("HAS_RISK_ASSESSMENT", "OUT", "风险评估")
        ],
        return_fields=[
            ReturnField("riskAssessments", "riskAssessments", required=False),
        ],
        enabled=True,
        required=False,
        description="风险评估（里程匹配）"
    ),
]


# ============================================================================
# 关联查询执行器
# ============================================================================

class AssociationQueryExecutor:
    """关联查询执行器"""

    def __init__(self, graph):
        """
        初始化执行器

        Args:
            graph: Neo4j图数据库连接
        """
        self.graph = graph
        self.plan_queries = [q for q in PLAN_BASED_QUERIES if q.enabled]
        self.mileage_queries = [q for q in MILEAGE_BASED_QUERIES if q.enabled]

    def execute_plan_queries(self, node_id: str) -> Dict[str, Any]:
        """
        执行基于方案ID的所有关联查询

        Args:
            node_id: 业务节点ID（node_id属性，如"node_004391"）

        Returns:
            查询结果字典 {query_name: result}
        """
        results = {}

        for query_config in self.plan_queries:
            try:
                if query_config.query_type == QueryType.DIRECT_PROPERTY:
                    result = self._execute_direct_property_query(node_id, query_config)
                elif query_config.query_type == QueryType.SINGLE_HOP:
                    result = self._execute_single_hop_query(node_id, query_config)
                elif query_config.query_type == QueryType.MULTI_HOP:
                    result = self._execute_multi_hop_query(node_id, query_config)
                else:
                    result = self._get_fallback_result(query_config)

                results[query_config.query_name] = result

            except Exception as e:
                if query_config.required:
                    raise
                results[query_config.query_name] = self._get_fallback_result(query_config)

        return results

    def execute_mileage_queries(self, line_name: str, mileage_start: float, mileage_end: float,
                               mileage_prefix: str, risk_type: str = None) -> Dict[str, Any]:
        """
        执行基于里程的所有关联查询

        Args:
            line_name: 线路名称
            mileage_start: 里程区间起始值（米）
            mileage_end: 里程区间终止值（米）
            mileage_prefix: 里程前缀（如"DK"）
            risk_type: 风险类型（可选）

        Returns:
            查询结果字典 {query_name: result}
        """
        results = {}

        for query_config in self.mileage_queries:
            try:
                if query_config.query_type == QueryType.MILEAGE_MATCH:
                    result = self._execute_mileage_match_query(
                        line_name, mileage_start, mileage_end, mileage_prefix, query_config
                    )
                elif query_config.query_type == QueryType.MULTI_HOP:
                    result = self._execute_mileage_multi_hop_query(
                        line_name, mileage_start, mileage_end, mileage_prefix, query_config
                    )
                else:
                    result = self._get_fallback_result(query_config)

                results[query_config.query_name] = result

            except Exception as e:
                if query_config.required:
                    raise
                results[query_config.query_name] = self._get_fallback_result(query_config)

        return results

    def _execute_direct_property_query(self, node_id: int,
                                       query_config: AssociationQuery) -> Any:
        """执行直接属性查询"""
        field_names = [f.field_name for f in query_config.return_fields]

        query = f"""
        MATCH (n:{query_config.source_node_type})
        WHERE n.node_id = $node_id
        RETURN {', '.join([f"n.{field} AS {field}" for field in field_names])}
        """

        logger.debug(f"[直接属性查询] {query_config.query_name}")
        logger.debug(f"查询语句: {query.strip()}")
        logger.debug(f"参数: node_id={node_id}")

        result = self.graph.run(query, node_id=node_id).data()

        logger.debug(f"原始结果: {result}")

        if not result:
            logger.warning(f"查询 {query_config.query_name} 返回空结果")
            return self._get_fallback_result(query_config)

        # 提取返回字段
        row = result[0]
        return {f.alias: row.get(f.field_name, f.fallback_value)
                for f in query_config.return_fields}

    def _execute_single_hop_query(self, node_id: int,
                                  query_config: AssociationQuery) -> List[Any]:
        """执行单跳关系查询"""
        step = query_config.query_path[0]
        field_names = [f.field_name for f in query_config.return_fields]

        # Cypher关系语法: -[关系类型]-> 或 <-[关系类型]-
        if step.direction == "OUT":
            relation_pattern = f"-[:{step.relation_type}]->"
        elif step.direction == "IN":
            relation_pattern = f"<-[:{step.relation_type}]-"
        else:  # BOTH
            relation_pattern = f"-[:{step.relation_type}]-"

        query = f"""
        MATCH (source:{query_config.source_node_type}){relation_pattern}(target:{step.target_node_type})
        WHERE source.node_id = $node_id
        RETURN {', '.join([f"target.{field} AS {field}" for field in field_names])}
        """

        logger.debug(f"[单跳查询] {query_config.query_name}")
        logger.debug(f"查询语句: {query.strip()}")
        logger.debug(f"参数: node_id={node_id}")

        results = self.graph.run(query, node_id=node_id).data()

        logger.debug(f"原始结果: {results}")

        if not results:
            logger.warning(f"查询 {query_config.query_name} 返回空结果")
            return []

        return [{f.alias: row.get(f.field_name, f.fallback_value)
                 for f in query_config.return_fields}
                for row in results]

    def _execute_multi_hop_query(self, node_id: int,
                                 query_config: AssociationQuery) -> List[Any]:
        """执行多跳路径查询（支持OPTIONAL MATCH）"""
        # 检查是否有optional步骤
        has_optional = any(step.optional for step in query_config.query_path)

        if not has_optional:
            # 原有逻辑：没有optional步骤，使用普通MATCH
            return self._execute_multi_hop_query_match(node_id, query_config)
        else:
            # 新逻辑：有optional步骤，使用OPTIONAL MATCH
            return self._execute_multi_hop_query_optional(node_id, query_config)

    def _execute_multi_hop_query_match(self, node_id: int,
                                       query_config: AssociationQuery) -> List[Any]:
        """执行普通多跳路径查询（无OPTIONAL MATCH）"""
        field_names = [f.field_name for f in query_config.return_fields]

        # 构建路径匹配
        path_pattern = f"(source:{query_config.source_node_type})"
        for i, step in enumerate(query_config.query_path):
            # Cypher关系语法: -[关系类型]-> 或 <-[关系类型]-
            if step.direction == "OUT":
                relation_pattern = f"-[:{step.relation_type}]->"
            elif step.direction == "IN":
                relation_pattern = f"<-[:{step.relation_type}]-"
            else:  # BOTH
                relation_pattern = f"-[:{step.relation_type}]-"
            path_pattern += f"{relation_pattern}(step{i}:{step.target_node_type})"

        query = f"""
        MATCH {path_pattern}
        WHERE source.node_id = $node_id
        RETURN {', '.join([f"step{len(query_config.query_path)-1}.{field} AS {field}"
                          for field in field_names])}
        """

        logger.debug(f"[多跳查询] {query_config.query_name}")
        logger.debug(f"查询语句: {query.strip()}")
        logger.debug(f"参数: node_id={node_id}")

        results = self.graph.run(query, node_id=node_id).data()

        logger.debug(f"原始结果: {results}")

        if not results:
            logger.warning(f"查询 {query_config.query_name} 返回空结果")
            return []

        return [{f.alias: row.get(f.field_name, f.fallback_value)
                 for f in query_config.return_fields}
                for row in results]

    def _execute_multi_hop_query_optional(self, node_id: int,
                                         query_config: AssociationQuery) -> List[Any]:
        """执行带OPTIONAL MATCH的多跳路径查询"""
        # 构建MATCH和OPTIONAL MATCH语句
        match_clauses = []
        where_conditions = []

        # 第一个关系使用MATCH
        if query_config.query_path:
            first_step = query_config.query_path[0]
            if first_step.direction == "OUT":
                relation_pattern = f"-[:{first_step.relation_type}]->"
            elif first_step.direction == "IN":
                relation_pattern = f"<-[:{first_step.relation_type}]-"
            else:  # BOTH
                relation_pattern = f"-[:{first_step.relation_type}]-"
            match_clauses.append(f"MATCH (source:{query_config.source_node_type}){relation_pattern}(step0:{first_step.target_node_type})")

        # 后续关系：如果是optional则使用OPTIONAL MATCH
        for i, step in enumerate(query_config.query_path[1:], start=1):
            if step.direction == "OUT":
                relation_pattern = f"-[:{step.relation_type}]->"
            elif step.direction == "IN":
                relation_pattern = f"<-[:{step.relation_type}]-"
            else:  # BOTH
                relation_pattern = f"-[:{step.relation_type}]-"

            if step.optional:
                match_clauses.append(f"OPTIONAL MATCH (step{i-1}){relation_pattern}(step{i}:{step.target_node_type})")
            else:
                match_clauses.append(f"MATCH (step{i-1}){relation_pattern}(step{i}:{step.target_node_type})")

        # 构建RETURN子句
        return_clauses = []
        for field in query_config.return_fields:
            # 确定字段来自哪个步骤
            if field.source_step >= 0:
                step_index = field.source_step
            else:
                step_index = len(query_config.query_path) - 1

            return_clauses.append(f"step{step_index}.{field.field_name} AS {field.field_name}")

        query = f"""
        {' '.join(match_clauses)}
        WHERE source.node_id = $node_id
        RETURN {', '.join(return_clauses)}
        """

        logger.debug(f"[多跳查询(OPTIONAL)] {query_config.query_name}")
        logger.debug(f"查询语句: {query.strip()}")
        logger.debug(f"参数: node_id={node_id}")

        results = self.graph.run(query, node_id=node_id).data()

        logger.debug(f"原始结果: {results}")

        if not results:
            logger.warning(f"查询 {query_config.query_name} 返回空结果")
            return []

        return [{f.alias: row.get(f.field_name, f.fallback_value)
                 for f in query_config.return_fields}
                for row in results]

    def _execute_mileage_match_query(self, line_name: str, mileage_start: float, mileage_end: float,
                                    mileage_prefix: str, query_config: AssociationQuery) -> List[Any]:
        """执行里程匹配查询（使用区间重叠判断）"""
        # 导入ChainageParser用于里程区间重叠判断
        from kg_construction.core.graph_inference.chainage_parser import ChainageParser

        field_names = [f.field_name for f in query_config.return_fields]

        # 避免重复返回mileage_field
        return_fields = []
        if query_config.mileage_field not in field_names:
            return_fields.append(f"n.{query_config.mileage_field} AS {query_config.mileage_field}")
        return_fields.extend([f"n.{field} AS {field}" for field in field_names])

        query = f"""
        MATCH (n:{query_config.source_node_type})
        WHERE n.{query_config.mileage_field} IS NOT NULL
        RETURN {', '.join(return_fields)}
        """

        logger.info(f"[里程匹配查询] {query_config.query_name}: 查询里程范围=[{mileage_start}, {mileage_end}]米, 前缀={mileage_prefix}")
        logger.debug(f"查询语句: {query.strip()}")

        results = self.graph.run(query).data()

        logger.debug(f"原始结果数: {len(results)}")

        if not results:
            return []

        # 在Python中进行里程区间重叠过滤
        filtered_results = []

        for record in results:
            chainage_str = record.get(query_config.mileage_field, "")
            if not chainage_str:
                continue

            try:
                # 使用ChainageParser解析节点的chainage字段
                node_range = ChainageParser.parse(chainage_str)

                if node_range:
                    # 使用用户输入的前缀构建查询区间（而不是节点的前缀）
                    query_range = (mileage_start, mileage_end, mileage_prefix)

                    if ChainageParser.overlaps(query_range, node_range):
                        # 提取返回字段
                        result = {f.alias: record.get(f.field_name, f.fallback_value)
                                for f in query_config.return_fields}
                        result["chainage"] = chainage_str  # 添加原始chainage信息
                        filtered_results.append(result)
                        logger.info(f"[里程匹配] ✅ 匹配成功: 查询{query_range} <-> 节点{node_range} ({chainage_str})")
                    else:
                        logger.debug(f"[里程匹配] ⚠️ 前缀不匹配或区间不重叠: 查询{query_range} <-> 节点{node_range} ({chainage_str})")
            except Exception as e:
                logger.warning(f"里程解析失败: {chainage_str}, 错误: {e}")
                continue

        logger.debug(f"过滤后结果数: {len(filtered_results)}")
        return filtered_results

    def _execute_mileage_multi_hop_query(self, line_name: str, mileage_start: float, mileage_end: float,
                                        mileage_prefix: str, query_config: AssociationQuery) -> List[Any]:
        """执行基于里程的多跳查询（使用区间重叠判断）"""
        # 导入ChainageParser用于里程区间重叠判断
        from kg_construction.core.graph_inference.chainage_parser import ChainageParser

        field_names = [f.field_name for f in query_config.return_fields]

        # 构建路径匹配（使用正确的Cypher关系语法）
        path_pattern = f"(source:{query_config.source_node_type})"
        for i, step in enumerate(query_config.query_path):
            # Cypher关系语法: -[关系类型]-> 或 <-[关系类型]-
            if step.direction == "OUT":
                relation_pattern = f"-[:{step.relation_type}]->"
            elif step.direction == "IN":
                relation_pattern = f"<-[:{step.relation_type}]-"
            else:  # BOTH
                relation_pattern = f"-[:{step.relation_type}]-"
            path_pattern += f"{relation_pattern}(step{i}:{step.target_node_type})"

        query = f"""
        MATCH {path_pattern}
        WHERE source.{query_config.mileage_field} IS NOT NULL
        RETURN source.{query_config.mileage_field} AS {query_config.mileage_field},
               {', '.join([f"step{len(query_config.query_path)-1}.{field} AS {field}"
                          for field in field_names])}
        """

        logger.debug(f"[里程多跳查询] {query_config.query_name}")
        logger.debug(f"查询语句: {query.strip()}")
        logger.debug(f"参数: mileage_range=[{mileage_start}, {mileage_end}], prefix={mileage_prefix}")

        results = self.graph.run(query).data()

        logger.debug(f"原始结果数: {len(results)}")

        if not results:
            return []

        # 在Python中进行里程区间重叠过滤
        filtered_results = []

        for record in results:
            chainage_str = record.get(query_config.mileage_field, "")
            if not chainage_str:
                continue

            try:
                # 使用ChainageParser解析节点的chainage字段
                node_range = ChainageParser.parse(chainage_str)

                if node_range:
                    # 使用用户输入的前缀构建查询区间（而不是节点的前缀）
                    query_range = (mileage_start, mileage_end, mileage_prefix)

                    if ChainageParser.overlaps(query_range, node_range):
                        # 提取返回字段
                        result = {f.alias: record.get(f.field_name, f.fallback_value)
                                for f in query_config.return_fields}
                        result["chainage"] = chainage_str  # 添加原始chainage信息
                        filtered_results.append(result)
                        logger.info(f"[里程匹配] ✅ 匹配成功: 查询{query_range} <-> 节点{node_range} ({chainage_str})")
                    else:
                        logger.debug(f"[里程匹配] ⚠️ 前缀不匹配或区间不重叠: 查询{query_range} <-> 节点{node_range} ({chainage_str})")
            except Exception as e:
                logger.warning(f"里程解析失败: {chainage_str}, 错误: {e}")
                continue

        logger.debug(f"过滤后结果数: {len(filtered_results)}")
        return filtered_results

        if not results:
            return []

        # 导入parse_mileage函数
        from retrieval.utils.kg_utils import parse_mileage

        # 在Python中进行里程范围过滤
        filtered_results = []
        for record in results:
            chainage_str = record.get(query_config.mileage_field, "")
            if not chainage_str:
                continue

            try:
                # 解析chainage字段（格式："起始里程～终止里程"）
                start, end = parse_mileage(chainage_str)
                # 检查目标里程是否在区间内
                if start <= mileage < end:
                    # 提取返回字段
                    result = {f.alias: record.get(f.field_name, f.fallback_value)
                            for f in query_config.return_fields}
                    result["chainage"] = chainage_str  # 添加原始chainage信息
                    filtered_results.append(result)
            except Exception:
                # 解析失败，跳过此节点
                continue

        return filtered_results

    def _get_fallback_result(self, query_config: AssociationQuery) -> Any:
        """获取回退值"""
        if query_config.query_type == QueryType.DIRECT_PROPERTY:
            return {f.alias: f.fallback_value
                   for f in query_config.return_fields}
        else:
            return []
