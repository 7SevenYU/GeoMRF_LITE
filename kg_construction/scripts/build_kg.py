import json
import logging
import sys
from pathlib import Path

# 智能检测项目根目录，支持两种运行方式
script_dir = Path(__file__).parent
if script_dir.name == "scripts":
    # 从项目根目录运行：python -m kg_construction.scripts.build_kg
    project_root = script_dir.parent.parent
else:
    # 在scripts目录直接运行：python build_kg.py（调试模式）
    project_root = script_dir.parent.parent

# 确保项目根目录在sys.path中
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from kg_construction.core.storage.neo4j_client import Neo4jClient
from kg_construction.core.storage.neo4j_sync_tracker import Neo4jSyncTracker
from kg_construction.core.storage.data_loader import DataLoader
from kg_construction.core.storage.graph_builder import GraphBuilder


class GraphDeduplication:
    """Neo4j图去重"""

    def __init__(self, neo4j_client: Neo4jClient):
        self.client = neo4j_client
        self.graph = neo4j_client.graph
        self.logger = logging.getLogger(__name__)

    def deduplicate_nodes(self, node_label: str, property_name: str):
        """根据属性值去重节点"""
        self.logger.info(f"开始去重 {node_label}，基于属性 {property_name}")

        # 查询所有重复的节点组
        query = f"""
        MATCH (n:{node_label})
        WHERE n.{property_name} IS NOT NULL
        WITH n.{property_name} as prop_value, collect(n) as nodes, count(*) as cnt
        WHERE cnt > 1
        RETURN prop_value, nodes, cnt
        ORDER BY cnt DESC
        """

        result = self.graph.run(query)

        total_duplicates = 0
        total_merged = 0

        for record in result:
            prop_value = record["prop_value"]
            nodes = record["nodes"]
            cnt = record["cnt"]

            total_duplicates += 1
            total_merged += cnt - 1

            self.logger.info(f"  {property_name}='{prop_value}': 找到 {cnt} 个重复节点")

            # 保留第一个节点（ID最小的）
            keep_node = nodes[0]
            keep_node_id = keep_node.identity

            # 处理其他节点
            for i in range(1, len(nodes)):
                dup_node = nodes[i]
                dup_node_id = dup_node.identity

                self.logger.info(f"    合并节点 {dup_node_id} -> {keep_node_id}")

                # 转移所有关系到保留的节点
                self._redirect_relationships(dup_node_id, keep_node_id)

                # 删除重复节点
                delete_query = """
                MATCH (dup)
                WHERE elementId(dup) = $dup_id
                DETACH DELETE dup
                """
                self.graph.run(delete_query, dup_id=dup_node_id)

        self.logger.info(f"{node_label} 去重完成: 找到 {total_duplicates} 组重复，合并了 {total_merged} 个节点")

    def _redirect_relationships(self, from_node_id: int, to_node_id: int):
        """转移一个节点的所有关系到另一个节点"""
        # 转移出边
        query_out = """
        MATCH (from)-[r]->(other)
        WHERE elementId(from) = $from_id
        RETURN type(r) as rel_type, properties(r) as props, elementId(other) as other_id
        """

        out_result = self.graph.run(query_out, from_id=from_node_id)
        for record in out_result:
            rel_type = record["rel_type"]
            props = record["props"]
            other_id = record["other_id"]

            create_out = f"""
            MATCH (to), (other)
            WHERE elementId(to) = $to_id AND elementId(other) = $other_id
            CREATE (to)-[r:{rel_type}]->(other)
            SET r += $props
            """
            self.graph.run(create_out, to_id=to_node_id, other_id=other_id, props=props)

        # 转移入边
        query_in = """
        MATCH (other)-[r]->(from)
        WHERE elementId(from) = $from_id
        RETURN type(r) as rel_type, properties(r) as props, elementId(other) as other_id
        """

        in_result = self.graph.run(query_in, from_id=from_node_id)
        for record in in_result:
            rel_type = record["rel_type"]
            props = record["props"]
            other_id = record["other_id"]

            create_in = f"""
            MATCH (other), (to)
            WHERE elementId(other) = $other_id AND elementId(to) = $to_id
            CREATE (other)-[r:{rel_type}]->(to)
            SET r += $props
            """
            self.graph.run(create_in, other_id=other_id, to_id=to_node_id, props=props)

    def deduplicate_all(self):
        """执行所有节点类型的去重"""
        dedup_config = {
            "围岩等级": "grade",
            "风险类型": "riskType",
            "预警等级": "warningGrade",
            "地质风险等级": "geologicalRiskGrade",
            "设计信息": "chainage"
        }

        self.logger.info("=" * 80)
        self.logger.info("开始Neo4j图去重")
        self.logger.info("=" * 80)

        for node_label, prop_name in dedup_config.items():
            try:
                self.deduplicate_nodes(node_label, prop_name)
            except Exception as e:
                self.logger.error(f"去重 {node_label} 失败: {e}")

        self.logger.info("=" * 80)
        self.logger.info("图去重完成")
        self.logger.info("=" * 80)


def setup_logging(log_level="INFO"):
    project_root = _get_project_root()
    log_dir = project_root / "kg_construction" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'build_kg.log', encoding='utf-8')
        ]
    )


def _create_singleton_nodes(neo4j_client):
    """
    预先创建所有单例节点（固定取值的节点类型）
    包括：围岩等级、风险类型、预警等级、地质风险等级
    如果已存在则跳过，避免重复创建
    """
    from py2neo import Node

    logger = logging.getLogger(__name__)
    logger.info("开始预创建单例节点...")

    # 定义所有单例节点的可能取值
    singleton_configs = {
        "围岩等级": [
            {"grade": "I"},
            {"grade": "II"},
            {"grade": "III"},
            {"grade": "IV"},
            {"grade": "V"},
            {"grade": "VI"}
        ],
        "风险类型": [
            {"riskType": "塌方"},
            {"riskType": "突涌"},
            {"riskType": "岩爆"},
            {"riskType": "大变形"},
            {"riskType": "掉块"},
            {"riskType": "富水破碎带"}
        ],
        "预警等级": [
            {"warningGrade": "Low"},
            {"warningGrade": "Middle"},
            {"warningGrade": "High"}
        ],
        "地质风险等级": [
            {"geologicalRiskGrade": "Low"},
            {"geologicalRiskGrade": "Middle"},
            {"geologicalRiskGrade": "High"},
            {"geologicalRiskGrade": "Critical"}
        ],
        "探测方法": [
            {"detectionMethod": "水平声波剖面"},
            {"detectionMethod": "TSP"},
            {"detectionMethod": "地质雷达"},
            {"detectionMethod": "超前钻探"},
            {"detectionMethod": "红外探水"},
            {"detectionMethod": "洞身纵向地质素描"}
        ]
    }

    created_count = 0
    skipped_count = 0

    for node_label, nodes_list in singleton_configs.items():
        for attributes in nodes_list:
            # 检查是否已存在（通过属性）
            attr_conditions = ", ".join([f"{k}: '{v}'" for k, v in attributes.items()])
            check_query = f"""
            MATCH (n:{node_label} {{{attr_conditions}}})
            RETURN count(n) as cnt
            """
            result = neo4j_client.graph.run(check_query)
            count = result.data()[0]["cnt"]

            if count == 0:
                # 不存在则创建
                node = Node(node_label, **attributes)
                neo4j_client.graph.create(node)
                created_count += 1
            else:
                skipped_count += 1

    logger.info(f"单例节点预创建完成，新创建 {created_count} 个，跳过 {skipped_count} 个已存在的节点")


def load_config(config_file: str) -> dict:
    # 将相对路径转换为基于project_root的绝对路径
    if not Path(config_file).is_absolute():
        config_file = project_root / config_file

    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("开始构建Neo4j知识图谱")
    logger.info("=" * 80)

    try:
        config = load_config("kg_construction/core/storage/neo4j_config.json")

        neo4j_client = Neo4jClient(
            uri=config["uri"],
            username=config["username"],
            password=config["password"],
            database=config.get("database", "neo4j")
        )

        neo4j_client.connect()

        # 预创建单例节点（固定取值的节点类型）
        _create_singleton_nodes(neo4j_client)

        sync_tracker = Neo4jSyncTracker()

        data_loader = DataLoader(
            extraction_results_dir="kg_construction/data/processed/extraction_results",
            extraction_mapping_file="kg_construction/core/extraction/extraction_mapping.json"
        )

        graph_builder = GraphBuilder(
            neo4j_client=neo4j_client,
            sync_tracker=sync_tracker,
            data_loader=data_loader,
            batch_size=config.get("batch_size", 100)
        )

        graph_builder.build_explicit_graph(force_rebuild=False)

        # 构建隐式关系（推理关系）
        graph_builder.build_implicit_graph()

        stats = graph_builder.get_statistics()
        logger.info("=" * 80)
        logger.info("知识图谱构建统计:")
        logger.info(f"  - 数据库节点总数: {stats['database']['node_count']}")
        logger.info(f"  - 数据库关系总数: {stats['database']['relation_count']}")
        logger.info(f"  - 已同步文档数量: {stats['synced_documents']}")
        logger.info(f"  - 已同步节点总数: {stats['total_nodes_synced']}")
        logger.info(f"  - 已同步关系总数: {stats['total_relations_synced']}")
        logger.info("=" * 80)

        neo4j_client.close()
        logger.info("知识图谱构建完成!")

    except Exception as e:
        logger.error(f"构建知识图谱时发生错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
