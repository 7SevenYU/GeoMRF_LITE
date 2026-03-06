"""
图推理配置：隐式关系链路配置

通过声明式配置定义节点间隐式关系的建立路径
"""

from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field


# ====================================================================
# 链路步骤类型定义
# ====================================================================

LinkType = Literal[
    "id_match",              # ID精确匹配
    "chainage_overlap",      # 里程区间重叠
    "chainage_contains",     # 里程区间包含（from节点包含to节点）
    "chainage_contained_by", # 里程区间被包含（from节点被to节点包含）
    "explicit_relation",     # 显式关系（已存在的关系）
    "attribute_match"        # 通用属性匹配
]


@dataclass
class LinkStep:
    """链路中的一个步骤"""

    # 步骤类型
    type: LinkType

    # 节点类型
    from_node: str        # 起始节点类型
    to_node: str          # 目标节点类型

    # 属性配置
    from_attribute: str   # 起始节点属性（用于里程计算或属性匹配）
    to_attribute: str     # 目标节点属性

    # 关系配置（如果这一步会生成关系）
    relation_type: Optional[str] = None      # 关系类型（可选）
    cypher_label: Optional[str] = None       # Cypher标签（可选）

    # 置信度
    confidence: float = 1.0

    # 额外参数
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationChain:
    """关系链路：定义如何建立两个节点间的隐式关系"""

    # 目标关系
    relation_type: str        # 最终的关系类型
    cypher_label: str         # Cypher标签

    # 链路起点和终点
    source_node_type: str     # 起始节点类型
    target_node_type: str     # 目标节点类型

    # 链路步骤（路径）
    path: List[LinkStep]

    # 配置
    enabled: bool = True
    description: str = ""
    confidence_threshold: float = 0.5  # 置信度阈值
    max_paths: int = 3  # 每个源节点最多建立的目标关系数量（用于多跳推理）


# ====================================================================
# 关系链路配置
# ====================================================================

RELATION_CHAINS: List[RelationChain] = [

    # ====================================================================
    # 链路1：应急响应措施 -> 历史处置案例
    # 方式：方案id精确匹配
    # ====================================================================
    RelationChain(
        relation_type="RESPONDS_TO",
        cypher_label="RESPONDS_TO",
        source_node_type="紧急响应措施",
        target_node_type="历史处置案例",
        path=[
            LinkStep(
                type="id_match",
                from_node="紧急响应措施",
                to_node="历史处置案例",
                from_attribute="s_id",
                to_attribute="s_id",
                relation_type="RESPONDS_TO",
                cypher_label="RESPONDS_TO",
                confidence=1.0
            )
        ],
        enabled=True,
        description="应急响应措施通过方案id关联到历史处置案例"
    ),


    # ====================================================================
    # 链路2：变更信息 -> 设计信息
    # 方式：里程区间重叠
    # ====================================================================
    RelationChain(
        relation_type="IS_ASSOCIATED_WITH",
        cypher_label="IS_ASSOCIATED_WITH",
        source_node_type="变更信息",
        target_node_type="设计信息",
        path=[
            LinkStep(
                type="chainage_overlap",
                from_node="变更信息",
                to_node="设计信息",
                from_attribute="chainage",
                to_attribute="chainage",
                relation_type="IS_ASSOCIATED_WITH",
                cypher_label="IS_ASSOCIATED_WITH",
                confidence=0.9
            )
        ],
        enabled=True,
        description="变更信息通过里程区间重叠关联到设计信息"
    ),


    # ====================================================================
    # 链路3：历史处置案例 -> 施工信息
    # 方式：里程区间重叠
    # ====================================================================
    RelationChain(
        relation_type="OCCURS_AT",
        cypher_label="OCCURS_AT",
        source_node_type="历史处置案例",
        target_node_type="施工信息",
        path=[
            LinkStep(
                type="chainage_overlap",
                from_node="历史处置案例",
                to_node="施工信息",
                from_attribute="chainage",
                to_attribute="chainage",
                relation_type="OCCURS_AT",
                cypher_label="OCCURS_AT",
                confidence=0.95
            )
        ],
        enabled=True,
        description="历史处置案例通过里程区间重叠关联到施工信息"
    ),


    # ====================================================================
    # 链路4：施工信息 -> 设计信息
    # 方式：里程区间被包含（施工区间被设计区间包含）
    # 注意：关系方向是 施工->设计，但包含关系是 设计包含施工
    # ====================================================================
    RelationChain(
        relation_type="IS_ASSOCIATED_WITH",
        cypher_label="IS_ASSOCIATED_WITH",
        source_node_type="施工信息",
        target_node_type="设计信息",
        path=[
            LinkStep(
                type="chainage_contained_by",
                from_node="施工信息",
                to_node="设计信息",
                from_attribute="chainage",
                to_attribute="chainage",
                relation_type="IS_ASSOCIATED_WITH",
                cypher_label="IS_ASSOCIATED_WITH",
                confidence=1.0
            )
        ],
        enabled=True,
        description="施工信息通过里程包含关联到设计信息（设计区间包含施工区间）"
    ),


    # ====================================================================
    # 链路5：应急响应措施 -> 探测结论 【多跳链路】
    # 路径：应急响应措施 -> 历史处置案例 -> 施工信息 -> 探测方法 -> 探测结论
    # ====================================================================
    RelationChain(
        relation_type="REFERS_TO",
        cypher_label="REFERS_TO",
        source_node_type="紧急响应措施",
        target_node_type="探测结论",
        path=[
            LinkStep(
                type="explicit_relation",
                from_node="紧急响应措施",
                to_node="历史处置案例",
                from_attribute="",
                to_attribute="",
                relation_type="RESPONDS_TO",
                confidence=1.0
            ),
            LinkStep(
                type="explicit_relation",
                from_node="历史处置案例",
                to_node="施工信息",
                from_attribute="",
                to_attribute="",
                relation_type="OCCURS_AT",
                confidence=0.95
            ),
            LinkStep(
                type="explicit_relation",
                from_node="施工信息",
                to_node="探测方法",
                from_attribute="",
                to_attribute="",
                relation_type="WAS_SURVEYED_BY",
                confidence=1.0
            ),
            LinkStep(
                type="explicit_relation",
                from_node="探测方法",
                to_node="探测结论",
                from_attribute="",
                to_attribute="",
                relation_type="INDICATES",
                confidence=1.0
            )
        ],
        enabled=True,
        max_paths=3,
        description="应急响应措施通过多跳路径关联到探测结论"
    ),


    # ====================================================================
    # 链路6：应急响应措施 -> 围岩等级 【多跳链路】
    # 路径：应急响应措施 -> 历史处置案例 -> 围岩等级
    # ====================================================================
    RelationChain(
        relation_type="CONSIDERS",
        cypher_label="CONSIDERS",
        source_node_type="紧急响应措施",
        target_node_type="围岩等级",
        path=[
            LinkStep(
                type="explicit_relation",
                from_node="紧急响应措施",
                to_node="历史处置案例",
                from_attribute="",
                to_attribute="",
                relation_type="RESPONDS_TO",
                confidence=1.0
            ),
            LinkStep(
                type="explicit_relation",
                from_node="历史处置案例",
                to_node="围岩等级",
                from_attribute="",
                to_attribute="",
                relation_type="HAS_SURROUNDING_ROCK_GRADE",
                confidence=1.0
            )
        ],
        enabled=True,
        max_paths=3,
        description="应急响应措施通过多跳路径关联到围岩等级"
    ),


    # ====================================================================
    # 链路7：应急响应措施 -> 地质风险等级 【多跳链路】
    # 路径：应急响应措施 -> 历史处置案例 -> 施工信息 -> 探测方法 -> 地质风险等级
    # ====================================================================
    RelationChain(
        relation_type="CONSIDERS",
        cypher_label="CONSIDERS",
        source_node_type="紧急响应措施",
        target_node_type="地质风险等级",
        path=[
            LinkStep(
                type="explicit_relation",
                from_node="紧急响应措施",
                to_node="历史处置案例",
                from_attribute="",
                to_attribute="",
                relation_type="RESPONDS_TO",
                confidence=1.0
            ),
            LinkStep(
                type="explicit_relation",
                from_node="历史处置案例",
                to_node="施工信息",
                from_attribute="",
                to_attribute="",
                relation_type="OCCURS_AT",
                confidence=0.95
            ),
            LinkStep(
                type="explicit_relation",
                from_node="施工信息",
                to_node="探测方法",
                from_attribute="",
                to_attribute="",
                relation_type="WAS_SURVEYED_BY",
                confidence=1.0
            ),
            LinkStep(
                type="explicit_relation",
                from_node="探测方法",
                to_node="地质风险等级",
                from_attribute="",
                to_attribute="",
                relation_type="INDICATES",
                confidence=1.0
            )
        ],
        enabled=True,
        max_paths=3,
        description="应急响应措施通过多跳路径关联到地质风险等级"
    ),


    # ====================================================================
    # 链路8：应急响应措施 -> 风险评估 【多跳链路】
    # 路径：应急响应措施 -> 历史处置案例 -> 施工信息 -> 设计信息 -> 风险评估
    # ====================================================================
    RelationChain(
        relation_type="CONSIDERS",
        cypher_label="CONSIDERS",
        source_node_type="紧急响应措施",
        target_node_type="风险评估",
        path=[
            LinkStep(
                type="explicit_relation",
                from_node="紧急响应措施",
                to_node="历史处置案例",
                from_attribute="",
                to_attribute="",
                relation_type="RESPONDS_TO",
                confidence=1.0
            ),
            LinkStep(
                type="explicit_relation",
                from_node="历史处置案例",
                to_node="施工信息",
                from_attribute="",
                to_attribute="",
                relation_type="OCCURS_AT",
                confidence=0.95
            ),
            LinkStep(
                type="explicit_relation",
                from_node="施工信息",
                to_node="设计信息",
                from_attribute="",
                to_attribute="",
                relation_type="IS_ASSOCIATED_WITH",
                confidence=1.0
            ),
            LinkStep(
                type="explicit_relation",
                from_node="设计信息",
                to_node="风险评估",
                from_attribute="",
                to_attribute="",
                relation_type="HAS_RISK_ASSESSMENT",
                confidence=1.0
            )
        ],
        enabled=True,
        max_paths=3,
        description="应急响应措施通过多跳路径关联到风险评估"
    ),


    # ====================================================================
    # 链路9：应急响应措施 -> 预警等级 【2跳链路】
    # 路径：应急响应措施 -> 历史处置案例 -> 预警等级
    # ====================================================================
    RelationChain(
        relation_type="CONSIDERS",
        cypher_label="CONSIDERS",
        source_node_type="紧急响应措施",
        target_node_type="预警等级",
        path=[
            LinkStep(
                type="explicit_relation",
                from_node="紧急响应措施",
                to_node="历史处置案例",
                from_attribute="",
                to_attribute="",
                relation_type="RESPONDS_TO",
                confidence=1.0
            ),
            LinkStep(
                type="explicit_relation",
                from_node="历史处置案例",
                to_node="预警等级",
                from_attribute="",
                to_attribute="",
                relation_type="HAS_WARNING_GRADE",
                confidence=1.0
            )
        ],
        enabled=True,
        max_paths=5,
        description="应急响应措施通过历史处置案例关联到预警等级"
    ),


    # ====================================================================
    # 链路10：应急响应措施 -> 施工规范
    # 方式：属性匹配（riskType）
    # ====================================================================
    RelationChain(
        relation_type="refersTo",
        cypher_label="REFERS_TO",
        source_node_type="紧急响应措施",
        target_node_type="施工规范",
        path=[
            LinkStep(
                type="attribute_match",
                from_node="紧急响应措施",
                to_node="施工规范",
                from_attribute="riskType",
                to_attribute="riskType",
                relation_type="REFERS_TO",
                cypher_label="REFERS_TO",
                confidence=1.0
            )
        ],
        enabled=True,
        description="应急响应措施通过风险类型匹配关联到施工规范（风险类型一致时建立关系）"
    ),
]


# ====================================================================
# 辅助函数
# ====================================================================

def get_enabled_chains() -> List[RelationChain]:
    """获取所有启用的链路"""
    return [chain for chain in RELATION_CHAINS if chain.enabled]


def get_chain_by_relation(relation_type: str) -> Optional[RelationChain]:
    """根据关系类型获取链路"""
    for chain in RELATION_CHAINS:
        if chain.relation_type == relation_type:
            return chain
    return None


def get_chains_by_source(node_type: str) -> List[RelationChain]:
    """根据起始节点类型获取所有链路"""
    return [chain for chain in RELATION_CHAINS
            if chain.source_node_type == node_type and chain.enabled]


def get_chains_by_target(node_type: str) -> List[RelationChain]:
    """根据目标节点类型获取所有链路"""
    return [chain for chain in RELATION_CHAINS
            if chain.target_node_type == node_type and chain.enabled]


def visualize_chain(chain: RelationChain) -> str:
    """可视化链路路径"""
    path_parts = [step.from_node for step in chain.path]
    path_parts.append(chain.target_node_type)
    path_str = " -> ".join(path_parts)
    return f"{chain.relation_type}: {path_str}"


def print_all_chains():
    """打印所有链路（用于调试）"""
    print("=" * 80)
    print("关系链路配置")
    print("=" * 80)

    for i, chain in enumerate(RELATION_CHAINS, 1):
        status = "✓ 启用" if chain.enabled else "✗ 禁用"
        print(f"\n{i}. {chain.relation_type} [{status}]")
        print(f"   描述: {chain.description}")
        print(f"   路径: {visualize_chain(chain)}")

        for j, step in enumerate(chain.path, 1):
            step_info = f"   步骤{j}: {step.from_node} --[{step.type}]--> {step.to_node}"
            if step.relation_type:
                step_info += f" (生成关系: {step.relation_type})"
            print(step_info)

    print("\n" + "=" * 80)
    print(f"总计: {len(RELATION_CHAINS)} 条链路, {len(get_enabled_chains())} 条启用")
    print("=" * 80)
