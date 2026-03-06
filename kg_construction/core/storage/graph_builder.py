import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from kg_construction.core.storage.neo4j_client import Neo4jClient
from kg_construction.core.storage.neo4j_sync_tracker import Neo4jSyncTracker
from kg_construction.core.storage.data_loader import DataLoader
from kg_construction.core.graph_inference.chainage_parser import ChainageParser


class GraphBuilder:
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        sync_tracker: Neo4jSyncTracker,
        data_loader: DataLoader,
        batch_size: int = 100
    ):
        self.neo4j_client = neo4j_client
        self.sync_tracker = sync_tracker
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

    def build_explicit_graph(self, force_rebuild: bool = False):
        if force_rebuild:
            self.logger.info("强制重建模式：清空现有数据")
            self.neo4j_client.clear_database()
            self.sync_tracker.reset_all()

        self.logger.info("开始构建显式知识图谱...")
        synced_files = set()

        if not force_rebuild:
            synced_files = set(self.sync_tracker.get_all_records().keys())
            self.logger.info(f"已同步文档数量: {len(synced_files)}")

        unsynced_results = self.data_loader.load_all_unsynced_results(synced_files)
        self.logger.info(f"待同步文档数量: {len(unsynced_results)}")

        if not unsynced_results:
            self.logger.info("没有需要同步的新文档")
            return

        all_nodes = []
        all_relations = []

        for source_file, document_type, nodes, relations in unsynced_results:
            all_nodes.extend(nodes)
            all_relations.extend(relations)
            self.logger.info(
                f"待处理文档: {Path(source_file).name} "
                f"({len(nodes)} 节点, {len(relations)} 关系)"
            )

        self.logger.info(f"总计待创建节点: {len(all_nodes)}")
        self.logger.info(f"总计待创建关系: {len(all_relations)}")

        # 创建节点ID到节点信息的映射
        node_info_map = {node["node_id"]: node for node in all_nodes}

        self._create_nodes_batch(all_nodes)
        self._create_relations_batch(all_relations, node_info_map)

        for source_file, document_type, nodes, relations in unsynced_results:
            self.sync_tracker.mark_synced(source_file, len(nodes), len(relations))
            self.logger.info(f"已同步: {Path(source_file).name}")

        db_info = self.neo4j_client.get_database_info()
        self.logger.info(f"显式图谱构建完成!")
        self.logger.info(f"数据库统计 - 节点总数: {db_info['node_count']}, 关系总数: {db_info['relation_count']}")

    def _create_nodes_batch(self, nodes: List[Dict]):
        self.logger.info("开始批量创建节点...")
        success_count = 0

        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i:i + self.batch_size]
            count = self.neo4j_client.create_nodes_batch(batch)
            success_count += count
            self.logger.info(f"节点批次 [{i//self.batch_size + 1}]: {count}/{len(batch)} 成功")

        self.logger.info(f"节点创建完成: {success_count}/{len(nodes)} 成功")

    def _create_relations_batch(self, relations: List[Dict], node_info_map: Dict[str, Dict]):
        self.logger.info("开始批量创建关系...")
        success_count = 0

        for i in range(0, len(relations), self.batch_size):
            batch = relations[i:i + self.batch_size]
            count = self.neo4j_client.create_relations_batch(batch, node_info_map)
            success_count += count
            self.logger.info(f"关系批次 [{i//self.batch_size + 1}]: {count}/{len(batch)} 成功")

        self.logger.info(f"关系创建完成: {success_count}/{len(relations)} 成功")
        return success_count

    def get_statistics(self) -> Dict:
        db_info = self.neo4j_client.get_database_info()
        sync_records = self.sync_tracker.get_all_records()

        return {
            "database": db_info,
            "synced_documents": len(sync_records),
            "total_nodes_synced": sum(r.get("node_count", 0) for r in sync_records.values()),
            "total_relations_synced": sum(r.get("relation_count", 0) for r in sync_records.values())
        }

    def build_implicit_graph(self) -> int:
        """
        构建隐式关系（推理关系）

        混合使用RelationInferrer（单跳）和Cypher查询（多跳）执行推理

        Returns:
            创建的隐式关系数量
        """
        from kg_construction.core.graph_inference.relation_inferrer import RelationInferrer
        from kg_construction.core.graph_inference.inference_config import get_enabled_chains

        self.logger.info("开始构建隐式关系...")

        # 从Neo4j加载所有节点（用于单跳推理）
        all_nodes = []
        query = """
        MATCH (n)
        RETURN
            n.node_id as node_id,
            labels(n)[0] as cypher_label,
            properties(n) as attributes
        """

        try:
            results = self.neo4j_client.execute_query(query)

            for result in results:
                node_data = {
                    "node_id": result["node_id"],
                    "node_type": result["cypher_label"],
                    "attributes": dict(result["attributes"])
                }
                all_nodes.append(node_data)

            self.logger.info(f"从Neo4j加载了 {len(all_nodes)} 个节点")

        except Exception as e:
            self.logger.error(f"从Neo4j加载节点失败: {e}")
            return 0

        # 分离单跳和多跳链路
        chains = get_enabled_chains()
        single_hop_chains = [chain for chain in chains if len(chain.path) == 1]
        multi_hop_chains = [chain for chain in chains if len(chain.path) >= 2]

        self.logger.info(f"单跳链路: {len(single_hop_chains)} 条, 多跳链路: {len(multi_hop_chains)} 条")

        # 1. 执行单跳推理（使用RelationInferrer）
        all_inferred_relations = []

        if single_hop_chains:
            self.logger.info("执行单跳推理...")
            inferrer = RelationInferrer(all_nodes, [])
            single_hop_relations = inferrer.infer_all()
            all_inferred_relations.extend(single_hop_relations)
            self.logger.info(f"单跳推理完成: {len(single_hop_relations)} 条关系")

            # 关键：先写入单跳推理结果到Neo4j
            if single_hop_relations:
                node_info_map = {node["node_id"]: node for node in all_nodes}
                single_hop_count = self._create_relations_batch(single_hop_relations, node_info_map)
                self.logger.info(f"单跳推理关系已写入Neo4j: {single_hop_count} 条")

        # 2. 执行多跳推理（使用Cypher查询，此时单跳结果已在Neo4j中）
        if multi_hop_chains:
            self.logger.info("执行多跳推理...")
            multi_hop_relations = self._infer_multi_hop_relations()
            all_inferred_relations.extend(multi_hop_relations)
            self.logger.info(f"多跳推理完成: {len(multi_hop_relations)} 条关系")

        # 3. 如果有多跳推理结果，写入Neo4j
        if multi_hop_chains and multi_hop_relations:
            node_info_map = {node["node_id"]: node for node in all_nodes}
            multi_hop_count = self._create_relations_batch(multi_hop_relations, node_info_map)
            self.logger.info(f"多跳推理关系已写入Neo4j: {multi_hop_count} 条")

        # 4. 统计所有推理关系（仅用于日志，不再重复创建）
        if all_inferred_relations:
            # 按关系类型统计
            from collections import defaultdict
            type_stats = defaultdict(lambda: {"source": set(), "target": set()})
            for rel in all_inferred_relations:
                rel_type = rel.get("relation_type", "unknown")
                source_id = rel.get("head_node_id")
                target_id = rel.get("tail_node_id")
                type_stats[rel_type]["source"].add(source_id)
                type_stats[rel_type]["target"].add(target_id)

            # 打印推理结果统计
            self.logger.info("=" * 60)
            self.logger.info("隐式关系推理结果统计:")
            for rel_type, stats in sorted(type_stats.items()):
                count = len(stats["source"])
                self.logger.info(f"  - {rel_type}: {count} 条关系")
            self.logger.info("=" * 60)

            # 返回总成功数
            total_success = 0
            if single_hop_chains and single_hop_relations:
                total_success += single_hop_count
            if multi_hop_chains and multi_hop_relations:
                total_success += multi_hop_count

            self.logger.info(f"隐式关系创建完成: {total_success}/{len(all_inferred_relations)} 成功")
            return total_success

        self.logger.info("没有推理出隐式关系")
        return 0

    def _infer_response_to_case(self) -> List[Dict[str, any]]:
        """
        推理1: 应急响应措施 → 历史处置案例 (方案id精确匹配)
        """
        relations = []

        # Cypher查询所有应急响应措施节点
        query = """
        MATCH (response:紧急响应措施)
        WHERE response.s_id IS NOT NULL
        RETURN response.node_id AS response_id, response.s_id AS plan_id
        """

        results = self.neo4j_client.execute_query(query)
        self.logger.info(f"找到 {len(results)} 个有s_id的应急响应措施")

        match_count = 0
        for result in results:
            response_id = result["response_id"]
            plan_id = result["plan_id"]

            # 查找匹配的历史处置案例
            case_query = """
            MATCH (case:历史处置案例)
            WHERE case.s_id = $plan_id
            RETURN case.node_id AS case_id
            """

            case_results = self.neo4j_client.execute_query(case_query, {"plan_id": plan_id})

            if case_results:
                match_count += 1
                self.logger.info(f"  s_id={plan_id}: 找到 {len(case_results)} 个匹配的历史处置案例")

            for case_result in case_results:
                case_id = case_result["case_id"]

                relations.append({
                    "relation_type": "RESPONDS_TO",
                    "cypher_label": "RESPONDS_TO",
                    "head_node_id": response_id,
                    "tail_node_id": case_id,
                    "confidence": 1.0,
                    "extraction_method": "inference_id_match",
                    "inferred": True
                })

        self.logger.info(f"RESPONDS_TO推理: {len(results)}个应急响应措施中, {match_count}个找到匹配, 共创建{len(relations)}条关系")
        return relations

    def _infer_change_to_design(self) -> List[Dict[str, any]]:
        """
        推理2: 变更信息 → 设计信息 (里程区间重叠)
        """
        relations = []

        # 查询所有变更信息节点
        change_query = """
        MATCH (change:变更信息)
        WHERE change.chainage IS NOT NULL
        RETURN change.node_id AS change_id, change.chainage AS change_chainage
        """

        change_results = self.neo4j_client.execute_query(change_query)

        # 查询所有设计信息节点
        design_query = """
        MATCH (design:设计信息)
        WHERE design.chainage IS NOT NULL
        RETURN design.node_id AS design_id, design.chainage AS design_chainage
        """

        design_results = self.neo4j_client.execute_query(design_query)

        # 构建设计节点列表
        design_list = [
            (r["design_id"], ChainageParser.parse(r["design_chainage"]))
            for r in design_results
            if ChainageParser.parse(r["design_chainage"])
        ]

        # 遍历变更信息，匹配设计信息
        for change_result in change_results:
            change_id = change_result["change_id"]
            change_range = ChainageParser.parse(change_result["change_chainage"])

            if not change_range:
                continue

            for design_id, design_range in design_list:
                if ChainageParser.overlaps(change_range, design_range):
                    relations.append({
                        "relation_type": "IS_ASSOCIATED_WITH",
                        "cypher_label": "IS_ASSOCIATED_WITH",
                        "head_node_id": change_id,
                        "tail_node_id": design_id,
                        "confidence": 0.9,
                        "extraction_method": "inference_chainage_overlap",
                        "inferred": True
                    })

        return relations

    def _infer_case_to_construction(self) -> List[Dict[str, any]]:
        """
        推理3: 历史处置案例 → 施工信息 (里程区间包含)
        """
        relations = []

        # 查询所有历史处置案例节点
        case_query = """
        MATCH (case:历史处置案例)
        WHERE case.分段位置 IS NOT NULL
        RETURN case.node_id AS case_id, case.分段位置 AS case_chainage
        """

        case_results = self.neo4j_client.execute_query(case_query)

        # 查询所有施工信息节点
        construction_query = """
        MATCH (construction:施工信息)
        WHERE construction.chainage IS NOT NULL
        RETURN construction.node_id AS construction_id, construction.chainage AS construction_chainage
        """

        construction_results = self.neo4j_client.execute_query(construction_query)

        # 构建施工节点列表
        construction_list = [
            (r["construction_id"], ChainageParser.parse(r["construction_chainage"]))
            for r in construction_results
            if ChainageParser.parse(r["construction_chainage"])
        ]

        # 遍历历史案例，匹配施工信息
        for case_result in case_results:
            case_id = case_result["case_id"]
            case_range = ChainageParser.parse(case_result["case_chainage"])

            if not case_range:
                continue

            for construction_id, construction_range in construction_list:
                if ChainageParser.contains_range(case_range, construction_range):
                    relations.append({
                        "relation_type": "OCCURS_AT",
                        "cypher_label": "OCCURS_AT",
                        "head_node_id": case_id,
                        "tail_node_id": construction_id,
                        "confidence": 0.95,
                        "extraction_method": "inference_chainage_contains",
                        "inferred": True
                    })

        return relations

    def _infer_construction_to_design(self) -> List[Dict[str, any]]:
        """
        推理4: 施工信息 → 设计信息 (里程被包含 - 设计区间包含施工区间)
        """
        relations = []

        # 查询所有施工信息节点
        construction_query = """
        MATCH (construction:施工信息)
        WHERE construction.chainage IS NOT NULL
        RETURN construction.node_id AS construction_id, construction.chainage AS construction_chainage
        """

        construction_results = self.neo4j_client.execute_query(construction_query)

        # 查询所有设计信息节点
        design_query = """
        MATCH (design:设计信息)
        WHERE design.chainage IS NOT NULL
        RETURN design.node_id AS design_id, design.chainage AS design_chainage
        """

        design_results = self.neo4j_client.execute_query(design_query)

        # 构建设计节点列表
        design_list = [
            (r["design_id"], ChainageParser.parse(r["design_chainage"]))
            for r in design_results
            if ChainageParser.parse(r["design_chainage"])
        ]

        # 遍历施工信息，匹配设计信息（设计包含施工）
        for construction_result in construction_results:
            construction_id = construction_result["construction_id"]
            construction_range = ChainageParser.parse(construction_result["construction_chainage"])

            if not construction_range:
                continue

            for design_id, design_range in design_list:
                # 设计区间包含施工区间
                if ChainageParser.contains_range(design_range, construction_range):
                    relations.append({
                        "relation_type": "IS_ASSOCIATED_WITH",
                        "cypher_label": "IS_ASSOCIATED_WITH",
                        "head_node_id": construction_id,
                        "tail_node_id": design_id,
                        "confidence": 1.0,
                        "extraction_method": "inference_chainage_contained_by",
                        "inferred": True
                    })

        return relations

    def _infer_multi_hop_relations(self) -> List[Dict[str, any]]:
        """
        使用Cypher查询执行多跳推理

        通过直接在Neo4j中执行多跳路径查询，提高推理效率

        Returns:
            推理出的关系列表
        """
        from kg_construction.core.graph_inference.inference_config import get_enabled_chains
        from kg_construction.core.graph_inference.cypher_query_builder import CypherQueryBuilder

        relations = []
        chains = get_enabled_chains()

        # 筛选出多跳链路（路径长度 >= 2）
        multi_hop_chains = [chain for chain in chains if len(chain.path) >= 2]

        if not multi_hop_chains:
            self.logger.info("没有启用的多跳推理链路")
            return relations

        self.logger.info(f"发现 {len(multi_hop_chains)} 条多跳推理链路")

        for chain in multi_hop_chains:
            try:
                # 生成Cypher查询
                query = CypherQueryBuilder.build_multi_hop_query(chain)

                # 调试：打印生成的Cypher查询
                self.logger.info(f"  链路: {chain.relation_type} ({chain.description})")
                self.logger.info(f"  Cypher查询:\n{query}")

                # 执行查询
                results = self.neo4j_client.execute_query(query)

                # 转换结果为关系格式
                for result in results:
                    relations.append({
                        "relation_type": result["relation_type"],
                        "cypher_label": result["cypher_label"],
                        "head_node_id": result["head_node_id"],
                        "tail_node_id": result["tail_node_id"],
                        "confidence": result["confidence"],
                        "extraction_method": f"inference_multi_hop_{chain.relation_type}",
                        "inferred": True
                    })

                self.logger.info(f"  - {chain.relation_type}: 推理出 {len(results)} 条关系")

            except Exception as e:
                self.logger.error(f"执行多跳推理 {chain.relation_type} 失败: {e}")

        return relations
