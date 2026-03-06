import argparse
import logging
import sys
import json
from pathlib import Path


def _get_project_root():
    """获取项目根目录"""
    script_dir = Path(__file__).parent
    if script_dir.name == "scripts":
        return script_dir.parent.parent
    else:
        return script_dir.parent.parent


def setup_logging(log_level="INFO"):
    """设置日志"""
    log_dir = Path(__file__).parent.parent.parent / "kg_construction" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'generate_embeddings.log', encoding='utf-8')
        ]
    )


def load_config(config_file: str) -> dict:
    """加载配置文件"""
    project_root = _get_project_root()

    if not Path(config_file).is_absolute():
        config_file = project_root / config_file

    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='为知识图谱节点生成向量嵌入')
    parser.add_argument(
        '--node-type',
        type=str,
        help='指定节点类型（如"紧急响应措施"），不指定则处理所有类型'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成向量（忽略缓存）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='批处理大小（覆盖配置文件中的设置）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='kg_construction/core/storage/neo4j_config.json',
        help='Neo4j配置文件路径'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("知识图谱向量嵌入生成工具")
    logger.info("=" * 80)

    try:
        # 确保项目根目录在sys.path中
        project_root = _get_project_root()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from kg_construction.core.storage.neo4j_client import Neo4jClient
        from kg_construction.core.embedding.embedding_manager import EmbeddingManager

        # 加载配置
        config = load_config(args.config)

        # 连接Neo4j
        neo4j_client = Neo4jClient(
            uri=config["uri"],
            username=config["username"],
            password=config["password"],
            database=config.get("database", "neo4j")
        )
        neo4j_client.connect()

        # 创建嵌入管理器
        embedding_mgr = EmbeddingManager(neo4j_client)

        # 如果指定了batch_size，更新配置
        if args.batch_size:
            embedding_mgr.config["batch_size"] = args.batch_size
            logger.info(f"使用自定义批处理大小: {args.batch_size}")

        # 生成向量
        if args.node_type:
            logger.info(f"为指定节点类型生成向量: {args.node_type}")

            # 从配置中查找该节点类型的vec_keys
            node_types_config = embedding_mgr.config.get("node_types", {})
            if args.node_type not in node_types_config:
                logger.error(f"节点类型 '{args.node_type}' 未在配置文件中定义")
                logger.info(f"已配置的节点类型: {list(node_types_config.keys())}")
                sys.exit(1)

            vec_keys = node_types_config[args.node_type].get("vec_keys", [])
            if not vec_keys:
                logger.error(f"节点类型 '{args.node_type}' 未配置vec_keys")
                sys.exit(1)

            embedding_mgr.generate_embeddings_for_node_type(
                args.node_type,
                vec_keys,
                force=args.force
            )
        else:
            logger.info("为所有配置的节点类型生成向量")
            embedding_mgr.generate_all_embeddings(force=args.force)

        # 关闭连接
        neo4j_client.close()

        logger.info("=" * 80)
        logger.info("向量嵌入生成完成!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"生成向量嵌入时发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
