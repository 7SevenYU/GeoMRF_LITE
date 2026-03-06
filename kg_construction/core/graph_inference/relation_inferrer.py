"""
关系推理器：基于配置文件推断节点间的隐式关系
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from kg_construction.core.graph_inference.chainage_parser import ChainageParser
from kg_construction.core.graph_inference.inference_config import get_enabled_chains, LinkType, RelationChain, LinkStep


class RelationInferrer:
    """基于配置的关系推理器"""

    def __init__(self, nodes: List[Dict[str, Any]], relations: List[Dict[str, Any]] = None):
        """
        Args:
            nodes: 节点列表，每个节点包含：
                - node_id
                - node_type
                - attributes
                - merge_keys
                - cypher_label
            relations: 显式关系列表（用于多跳推理）
        """
        self.nodes = nodes
        self.relations = relations or []
        self.nodes_by_type = self._group_by_type()
        self.relations_by_nodes = self._index_relations() if relations else {}

    def _group_by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        """按node_type分组节点，提升查询性能"""
        grouped = defaultdict(list)
        for node in self.nodes:
            node_type = node.get("node_type")
            if node_type:
                grouped[node_type].append(node)
        return dict(grouped)

    def infer_all(self) -> List[Dict[str, Any]]:
        """
        执行所有推理，返回关系列表

        从配置文件读取启用的链路，逐个执行

        Returns:
            推理出的关系列表
        """
        all_relations = []

        # 从配置获取所有启用的链路
        chains = get_enabled_chains()

        # 执行每个链路
        for chain in chains:
            chain_relations = self._execute_chain(chain)
            all_relations.extend(chain_relations)

        return all_relations

    def _index_relations(self) -> Dict[str, List[Dict[str, Any]]]:
        """索引显式关系，用于多跳推理

        Returns:
            {node_id: [relations]} 字典
        """
        index = defaultdict(list)
        for rel in self.relations:
            head_id = rel.get("head_node_id")
            tail_id = rel.get("tail_node_id")
            if head_id:
                index[head_id].append(rel)
            if tail_id:
                index[tail_id].append(rel)
        return dict(index)

    def _execute_chain(self, chain: RelationChain) -> List[Dict[str, Any]]:
        """
        执行单个推理链路

        Args:
            chain: 关系链路配置

        Returns:
            推理出的关系列表
        """
        # 单跳链路：直接执行推理
        if len(chain.path) == 1:
            return self._execute_single_hop(chain)

        # 多跳链路：逐步执行
        return self._execute_multi_hop(chain)

    def _execute_single_hop(self, chain: RelationChain) -> List[Dict[str, Any]]:
        """执行单跳推理"""
        step = chain.path[0]
        relations = []

        source_nodes = self.nodes_by_type.get(step.from_node, [])
        target_nodes = self.nodes_by_type.get(step.to_node, [])

        for source in source_nodes:
            for target in target_nodes:
                if self._check_match(source, target, step):
                    relation = {
                        "relation_type": chain.relation_type,
                        "cypher_label": chain.cypher_label,
                        "head_node_id": source["node_id"],
                        "tail_node_id": target["node_id"],
                        "head_merge_key": self._get_first_merge_key(source),
                        "tail_merge_key": self._get_first_merge_key(target),
                        "inferred": True,
                        "confidence": step.confidence,
                        "evidence": self._generate_single_hop_evidence(source, target, step),
                        "inference_method": step.type
                    }
                    relations.append(relation)

        return relations

    def _execute_link_step(self, step: LinkStep, chain: RelationChain) -> List[Dict[str, Any]]:
        """执行单个链路步骤（向后兼容的入口）"""
        return self._execute_single_hop(chain)

    def _generate_single_hop_evidence(self, source: Dict[str, Any], target: Dict[str, Any], step: LinkStep) -> str:
        """生成单跳推理的证据字符串"""
        if step.type == "id_match":
            source_value = source.get("attributes", {}).get(step.from_attribute)
            return f"{step.from_attribute}匹配: {source_value}"
        elif step.type in ["chainage_overlap", "chainage_contains"]:
            source_value = source.get("attributes", {}).get(step.from_attribute)
            target_value = target.get("attributes", {}).get(step.to_attribute)
            if step.type == "chainage_overlap":
                return f"里程重叠: {source_value} ∩ {target_value}"
            else:
                return f"里程包含: {source_value} 包含 {target_value}"
        elif step.type == "explicit_relation":
            return f"显式关系: {step.relation_type}"
        elif step.type == "attribute_match":
            source_value = source.get("attributes", {}).get(step.from_attribute)
            target_value = target.get("attributes", {}).get(step.to_attribute)
            return f"属性匹配: {step.from_attribute}={source_value} == {step.to_attribute}={target_value}"
        else:
            return f"{step.type}"

    def _execute_multi_hop(self, chain: RelationChain) -> List[Dict[str, Any]]:
        """
        执行多跳推理

        通过逐步遍历链路，找到从源节点到目标节点的所有路径
        """
        relations = []

        # 获取源节点
        source_nodes = self.nodes_by_type.get(chain.source_node_type, [])

        for source_node in source_nodes:
            # 执行链路，找到所有可能的路径
            paths = self._find_paths(source_node, chain)

            # 为每条路径生成关系
            for path in paths:
                relation = self._create_relation_from_path(
                    source_node, path, chain
                )
                if relation:
                    relations.append(relation)

        return relations

    def _find_paths(self, start_node: Dict[str, Any], chain: RelationChain) -> List[List[Dict[str, Any]]]:
        """
        查找从起始节点到目标节点的所有路径

        使用深度优先搜索遍历链路

        Args:
            start_node: 起始节点
            chain: 关系链路配置

        Returns:
            路径列表，每条路径是节点序列
        """
        paths = []
        self._dfs_find_paths(
            current_node=start_node,
            chain=chain,
            step_index=0,
            current_path=[start_node],
            paths=paths
        )
        return paths

    def _dfs_find_paths(
        self,
        current_node: Dict[str, Any],
        chain: RelationChain,
        step_index: int,
        current_path: List[Dict[str, Any]],
        paths: List[List[Dict[str, Any]]]
    ):
        """深度优先搜索路径"""
        # 基本情况：已到达链路末尾
        if step_index >= len(chain.path):
            # 检查最后一个节点是否是目标类型
            if current_node.get("node_type") == chain.target_node_type:
                paths.append(current_path[:])
            return

        step = chain.path[step_index]
        next_nodes = self._find_next_nodes(current_node, step)

        for next_node in next_nodes:
            # 避免循环
            if next_node["node_id"] in [n["node_id"] for n in current_path]:
                continue

            current_path.append(next_node)
            self._dfs_find_paths(next_node, chain, step_index + 1, current_path, paths)
            current_path.pop()

    def _find_next_nodes(self, current_node: Dict[str, Any], step: LinkStep) -> List[Dict[str, Any]]:
        """
        根据链路步骤找到下一个节点

        Args:
            current_node: 当前节点
            step: 链路步骤配置

        Returns:
            符合条件的下一个节点列表
        """
        target_type = step.to_node
        candidates = self.nodes_by_type.get(target_type, [])

        matched = []

        for candidate in candidates:
            if self._check_match(current_node, candidate, step):
                matched.append(candidate)

        return matched

    def _check_match(self, source: Dict[str, Any], target: Dict[str, Any], step: LinkStep) -> bool:
        """
        检查两个节点是否满足链路步骤的匹配条件

        Args:
            source: 源节点
            target: 目标节点
            step: 链路步骤配置

        Returns:
            True if 匹配, False otherwise
        """
        if step.type == "id_match":
            return self._check_id_match(source, target, step)
        elif step.type == "chainage_overlap":
            return self._check_chainage_overlap(source, target, step)
        elif step.type == "chainage_contains":
            return self._check_chainage_contains(source, target, step)
        elif step.type == "chainage_contained_by":
            return self._check_chainage_contained_by(source, target, step)
        elif step.type == "explicit_relation":
            return self._check_explicit_relation(source, target, step)
        elif step.type == "attribute_match":
            return self._check_attribute_match(source, target, step)
        else:
            return False

    def _check_id_match(self, source: Dict[str, Any], target: Dict[str, Any], step: LinkStep) -> bool:
        """检查ID精确匹配"""
        source_value = source.get("attributes", {}).get(step.from_attribute)
        target_value = target.get("attributes", {}).get(step.to_attribute)

        return source_value is not None and source_value == target_value

    def _check_chainage_overlap(self, source: Dict[str, Any], target: Dict[str, Any], step: LinkStep) -> bool:
        """检查里程区间重叠"""
        source_chainage = source.get("attributes", {}).get(step.from_attribute)
        target_chainage = target.get("attributes", {}).get(step.to_attribute)

        if not source_chainage or not target_chainage:
            return False

        source_range = ChainageParser.parse(source_chainage)
        target_range = ChainageParser.parse(target_chainage)

        if not source_range or not target_range:
            return False

        return ChainageParser.overlaps(source_range, target_range)

    def _check_chainage_contains(self, source: Dict[str, Any], target: Dict[str, Any], step: LinkStep) -> bool:
        """检查里程包含（源节点区间包含目标节点区间）"""
        source_chainage = source.get("attributes", {}).get(step.from_attribute)
        target_chainage = target.get("attributes", {}).get(step.to_attribute)

        if not source_chainage or not target_chainage:
            return False

        source_range = ChainageParser.parse(source_chainage)
        target_range = ChainageParser.parse(target_chainage)

        if not source_range or not target_range:
            return False

        return ChainageParser.contains_range(source_range, target_range)

    def _check_chainage_contained_by(self, source: Dict[str, Any], target: Dict[str, Any], step: LinkStep) -> bool:
        """检查里程被包含（源节点区间被目标节点区间包含）"""
        source_chainage = source.get("attributes", {}).get(step.from_attribute)
        target_chainage = target.get("attributes", {}).get(step.to_attribute)

        if not source_chainage or not target_chainage:
            return False

        source_range = ChainageParser.parse(source_chainage)
        target_range = ChainageParser.parse(target_chainage)

        if not source_range or not target_range:
            return False

        # source被target包含 = target包含source
        return ChainageParser.contains_range(target_range, source_range)

    def _check_explicit_relation(self, source: Dict[str, Any], target: Dict[str, Any], step: LinkStep) -> bool:
        """检查是否存在显式关系"""
        source_id = source.get("node_id")
        target_id = target.get("node_id")

        # 查找从source到target的显式关系
        for rel in self.relations:
            if (rel.get("head_node_id") == source_id and
                rel.get("tail_node_id") == target_id and
                rel.get("relation_type") == step.relation_type):
                return True

        return False

    def _check_attribute_match(self, source: Dict[str, Any], target: Dict[str, Any], step: LinkStep) -> bool:
        """
        检查属性值匹配

        Args:
            source: 源节点
            target: 目标节点
            step: 链路步骤配置（包含from_attribute和to_attribute）

        Returns:
            True if 属性值匹配且都不为None, False otherwise
        """
        source_value = source.get("attributes", {}).get(step.from_attribute)
        target_value = target.get("attributes", {}).get(step.to_attribute)

        # 两个值都必须存在且相等
        return source_value is not None and source_value == target_value

    def _create_relation_from_path(
        self,
        source_node: Dict[str, Any],
        path: List[Dict[str, Any]],
        chain: RelationChain
    ) -> Optional[Dict[str, Any]]:
        """
        从路径创建关系

        Args:
            source_node: 起始节点
            path: 节点路径
            chain: 关系链路配置

        Returns:
            关系字典或None
        """
        if len(path) < 2:
            return None

        target_node = path[-1]

        # 计算路径置信度
        confidence = self._calculate_path_confidence(chain)

        # 生成证据字符串
        evidence = self._generate_evidence(path, chain)

        relation = {
            "relation_type": chain.relation_type,
            "cypher_label": chain.cypher_label,
            "head_node_id": source_node["node_id"],
            "tail_node_id": target_node["node_id"],
            "head_merge_key": self._get_first_merge_key(source_node),
            "tail_merge_key": self._get_first_merge_key(target_node),
            "inferred": True,
            "confidence": confidence,
            "evidence": evidence,
            "inference_method": f"multi_hop_{len(chain.path)}_steps"
        }

        return relation

    def _calculate_path_confidence(self, chain: RelationChain) -> float:
        """
        计算路径置信度

        单跳：使用步骤置信度
        多跳：所有步骤置信度相乘
        """
        confidences = [step.confidence for step in chain.path]

        if len(confidences) == 1:
            return confidences[0]

        # 多跳：置信度相乘
        product = 1.0
        for conf in confidences:
            product *= conf

        return round(product, 4)

    def _generate_evidence(self, path: List[Dict[str, Any]], chain: RelationChain) -> str:
        """生成推理证据字符串"""
        if len(path) <= 2:
            # 单跳
            step = chain.path[0]
            return f"通过{step.type}关联: {step.from_node} -> {step.to_node}"
        else:
            # 多跳
            node_names = " -> ".join([p.get("node_type", "") for p in path])
            return f"多跳路径推理: {node_names}"

    def _infer_change_design(self) -> List[Dict[str, Any]]:
        """
        推理1: 变更信息 → 设计信息（里程重叠）

        逻辑：
        - 遍历所有变更信息节点和设计信息节点
        - 解析两者的chainage属性
        - 如果里程区间重叠，创建isAssociatedWith关系

        Returns:
            推理出的关系列表
        """
        relations = []

        change_nodes = self.nodes_by_type.get("变更信息", [])
        design_nodes = self.nodes_by_type.get("设计信息", [])

        for change in change_nodes:
            change_chainage = change.get("attributes", {}).get("chainage")
            if not change_chainage:
                continue

            change_range = ChainageParser.parse(change_chainage)
            if not change_range:
                continue

            for design in design_nodes:
                design_chainage = design.get("attributes", {}).get("chainage")
                if not design_chainage:
                    continue

                design_range = ChainageParser.parse(design_chainage)
                if not design_range:
                    continue

                # 判断里程区间是否重叠
                if ChainageParser.overlaps(change_range, design_range):
                    relation = {
                        "relation_type": "IS_ASSOCIATED_WITH",
                        "cypher_label": "IS_ASSOCIATED_WITH",
                        "head_node_id": change["node_id"],
                        "tail_node_id": design["node_id"],
                        "head_merge_key": self._get_first_merge_key(change),
                        "tail_merge_key": self._get_first_merge_key(design),
                        "inferred": True,
                        "confidence": 0.9,
                        "evidence": f"里程重叠: {change_chainage} ∩ {design_chainage}",
                        "inference_method": "chainage_overlap"
                    }
                    relations.append(relation)

        return relations

    def _infer_case_construction(self) -> List[Dict[str, Any]]:
        """
        推理2: 历史处置案例 → 施工信息（里程重叠）

        逻辑：
        - 历史案例有"chainage"："DK13+250.00~DK13+274.00"（区间）
        - 施工信息有"chainage"："DK13+260~DK13+280"（区间）
        - 判断两个区间是否重叠

        Returns:
            推理出的关系列表
        """
        relations = []

        case_nodes = self.nodes_by_type.get("历史处置案例", [])
        construction_nodes = self.nodes_by_type.get("施工信息", [])

        for case in case_nodes:
            case_chainage = case.get("attributes", {}).get("chainage")
            if not case_chainage:
                continue

            case_range = ChainageParser.parse(case_chainage)
            if not case_range:
                continue

            for construction in construction_nodes:
                construction_chainage = construction.get("attributes", {}).get("chainage")
                if not construction_chainage:
                    continue

                construction_range = ChainageParser.parse(construction_chainage)
                if not construction_range:
                    continue

                # 判断两个里程区间是否重叠
                if ChainageParser.overlaps(case_range, construction_range):
                    relation = {
                        "relation_type": "OCCURS_AT",
                        "cypher_label": "OCCURS_AT",
                        "head_node_id": case["node_id"],
                        "tail_node_id": construction["node_id"],
                        "head_merge_key": self._get_first_merge_key(case),
                        "tail_merge_key": self._get_first_merge_key(construction),
                        "inferred": True,
                        "confidence": 0.95,
                        "evidence": f"里程重叠: {case_chainage} ∩ {construction_chainage}",
                        "inference_method": "chainage_overlap"
                    }
                    relations.append(relation)

        return relations

    def _infer_response_case(self) -> List[Dict[str, Any]]:
        """
        推理3: 应急响应措施 → 历史处置案例（方案id匹配）

        逻辑：
        - 应急措施有"s_id"：2
        - 历史案例有"s_id"：2
        - 精确匹配id值

        Returns:
            推理出的关系列表
        """
        relations = []

        response_nodes = self.nodes_by_type.get("紧急响应措施", [])
        case_nodes = self.nodes_by_type.get("历史处置案例", [])

        for response in response_nodes:
            response_plan_id = response.get("attributes", {}).get("s_id")
            if response_plan_id is None:
                continue

            for case in case_nodes:
                case_plan_id = case.get("attributes", {}).get("s_id")
                if case_plan_id is None:
                    continue

                # 精确匹配方案id
                if response_plan_id == case_plan_id:
                    relation = {
                        "relation_type": "RESPONDS_TO",
                        "cypher_label": "RESPONDS_TO",
                        "head_node_id": response["node_id"],
                        "tail_node_id": case["node_id"],
                        "head_merge_key": self._get_first_merge_key(response),
                        "tail_merge_key": self._get_first_merge_key(case),
                        "inferred": True,
                        "confidence": 1.0,
                        "evidence": f"方案id匹配: {response_plan_id}",
                        "inference_method": "exact_match"
                    }
                    relations.append(relation)

        return relations

    def _get_first_merge_key(self, node: Dict[str, Any]) -> str:
        """
        获取节点的第一个merge_key

        Args:
            node: 节点字典

        Returns:
            第一个merge_key字符串
        """
        merge_keys = node.get("merge_keys", [])
        if merge_keys:
            return merge_keys[0]
        return node.get("cypher_label", "")
