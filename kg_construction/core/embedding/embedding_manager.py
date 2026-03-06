import os
import json
import torch
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from py2neo import Graph


def _get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent.parent


class EmbeddingManager:
    """向量嵌入管理器，为知识图谱节点生成向量嵌入"""

    def __init__(self, neo4j_client, config_file: Optional[str] = None):
        """
        初始化嵌入管理器

        Args:
            neo4j_client: Neo4j客户端对象
            config_file: 配置文件路径，默认为embedding_config.json
        """
        self.neo4j_client = neo4j_client
        self.graph = neo4j_client.graph
        self.logger = logging.getLogger(__name__)

        # 加载配置
        if config_file is None:
            project_root = _get_project_root()
            config_file = project_root / "kg_construction" / "core" / "embedding" / "embedding_config.json"

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 初始化模型
        self.bge_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.use_bge = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_model()

        # 加载缓存
        self.cache_file = self.config.get("cache_file", "kg_construction/data/processed/embedding_cache.json")
        if not Path(self.cache_file).is_absolute():
            project_root = _get_project_root()
            self.cache_file = project_root / self.cache_file

        self.embeddings_cache = self._load_cache()

    def _init_model(self):
        """初始化嵌入模型（优先BGE，失败回退BERT）"""
        model_type = self.config.get("model_type", "bge")
        project_root = _get_project_root()

        if model_type == "bge":
            try:
                from sentence_transformers import SentenceTransformer
                bge_model_path = self.config.get("bge_model_path")

                if bge_model_path:
                    # 转换为绝对路径
                    if not Path(bge_model_path).is_absolute():
                        full_bge_path = project_root / bge_model_path
                    else:
                        full_bge_path = Path(bge_model_path)

                    self.bge_model = SentenceTransformer(str(full_bge_path))
                    self.logger.info(f"从本地路径加载BGE模型: {full_bge_path}")
                else:
                    self.bge_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
                    self.logger.info("从HuggingFace加载BGE模型: BAAI/bge-large-zh-v1.5")

                self.use_bge = True
                return
            except ImportError:
                self.logger.warning("sentence-transformers未安装，回退到BERT模型")
            except Exception as e:
                self.logger.warning(f"BGE模型加载失败: {e}，回退到BERT模型")

        # 回退到BERT模型
        try:
            from transformers import BertTokenizer, BertModel
            bert_model_path = self.config.get("bert_model_path")

            if bert_model_path:
                # 转换为绝对路径
                if not Path(bert_model_path).is_absolute():
                    full_bert_path = project_root / bert_model_path
                else:
                    full_bert_path = Path(bert_model_path)
            else:
                full_bert_path = project_root / "models" / "bert-base-chinese"

            self.bert_tokenizer = BertTokenizer.from_pretrained(str(full_bert_path))
            self.bert_model = BertModel.from_pretrained(str(full_bert_path))
            self.bert_model.to(self.device)
            self.bert_model.eval()
            self.logger.info(f"成功加载BERT模型: {full_bert_path}")
        except Exception as e:
            self.logger.error(f"BERT模型加载失败: {e}")
            raise

    def _load_cache(self) -> Dict[str, bool]:
        """加载嵌入缓存"""
        if Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"加载缓存文件失败: {e}")
        return {}

    def _save_cache(self):
        """保存嵌入缓存"""
        try:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.embeddings_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存缓存文件失败: {e}")

    def get_text_embedding(self, text: str):
        """
        生成文本的向量嵌入

        Args:
            text: 输入文本

        Returns:
            向量数组
        """
        if not text or not text.strip():
            self.logger.warning("输入文本为空，返回零向量")
            if self.use_bge:
                return None
            else:
                return None

        try:
            if self.use_bge:
                # 使用BGE模型
                embedding = self.bge_model.encode([text], normalize_embeddings=True)[0]
                return embedding
            else:
                # 使用BERT模型
                inputs = self.bert_tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                return embedding.cpu().numpy()
        except Exception as e:
            self.logger.error(f"生成文本向量失败: {e}")
            return None

    def build_node_description(self, node: Dict[str, Any], vec_keys: List[str]) -> str:
        """
        从节点属性构建文本描述

        Args:
            node: 节点数据字典
            vec_keys: 需要包含的属性键列表

        Returns:
            拼接后的文本描述
        """
        description_parts = []

        for key in vec_keys:
            value = node.get(key)

            if value is None:
                continue

            # 特殊处理keywords字段（数组类型）
            if key == "keywords":
                if isinstance(value, list):
                    keywords_text = " ".join(str(k) for k in value if k)
                    if keywords_text:
                        description_parts.append(keywords_text)
                else:
                    description_parts.append(str(value))
            else:
                description_parts.append(str(value))

        return " ".join(description_parts).strip()

    def process_batch(self, node_ids: List[str], texts: List[str]):
        """
        批量处理节点，生成向量并写入Neo4j

        Args:
            node_ids: 节点ID列表
            texts: 文本描述列表
        """
        vector_field = self.config.get("vector_field", "embedding_vector")
        batch_size = self.config.get("batch_size", 32)

        # 批量生成向量
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            self.logger.info(f"  处理批次 {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            for text in batch_texts:
                embedding = self.get_text_embedding(text)
                embeddings.append(embedding)

        # 写入Neo4j
        success_count = 0
        for node_id, embedding in zip(node_ids, embeddings):
            if embedding is None:
                self.logger.warning(f"节点 {node_id} 向量生成失败，跳过")
                continue

            try:
                # 将向量转换为字符串格式
                vector_str = ','.join(f'{x:.6f}' for x in embedding)

                # 更新节点
                query = f"""
                MATCH (n)
                WHERE elementId(n) = $node_id
                SET n.{vector_field} = [$vector_str]
                """
                self.graph.run(query, node_id=node_id, vector_str=vector_str)

                # 记录到缓存
                self.embeddings_cache[node_id] = True
                success_count += 1

            except Exception as e:
                self.logger.error(f"保存节点 {node_id} 向量失败: {e}")

        self.logger.info(f"  批次完成: 成功 {success_count}/{len(node_ids)}")

    def generate_embeddings_for_node_type(
        self,
        node_label: str,
        vec_keys: List[str],
        force: bool = False
    ):
        """
        为指定类型的所有节点生成向量嵌入

        Args:
            node_label: 节点类型标签（如"紧急响应措施"）
            vec_keys: 用于构建向量的属性键列表
            force: 是否强制重新生成（忽略缓存）
        """
        self.logger.info(f"开始为节点类型 '{node_label}' 生成向量嵌入")

        vector_field = self.config.get("vector_field", "embedding_vector")
        batch_size = self.config.get("batch_size", 32)

        # 查询所有该类型的节点
        query = f"""
        MATCH (n:{node_label})
        RETURN elementId(n) AS nodeId, n AS node
        """

        try:
            results = self.graph.run(query).data()
        except Exception as e:
            self.logger.error(f"查询节点失败: {e}")
            return

        if not results:
            self.logger.warning(f"未找到类型为 '{node_label}' 的节点")
            return

        self.logger.info(f"找到 {len(results)} 个 '{node_label}' 节点")

        # 过滤已处理的节点（除非force=True）
        nodes_to_process = []
        for record in results:
            node_id = record["nodeId"]
            node = dict(record["node"])

            # 检查是否已存在向量
            if not force and node_id in self.embeddings_cache:
                continue

            # 检查节点是否已有向量字段
            if not force and vector_field in node and node[vector_field] is not None:
                self.embeddings_cache[node_id] = True
                continue

            nodes_to_process.append((node_id, node))

        self.logger.info(f"需要处理 {len(nodes_to_process)} 个节点（已跳过已有向量的节点）")

        if not nodes_to_process:
            self.logger.info("所有节点已有向量，无需生成")
            return

        # 批量处理
        texts, node_ids = [], []
        for node_id, node in nodes_to_process:
            desc = self.build_node_description(node, vec_keys)
            if not desc:
                self.logger.warning(f"节点 {node_id} 构建的描述为空，跳过")
                continue

            texts.append(desc)
            node_ids.append(node_id)

            if len(texts) >= batch_size:
                self.process_batch(node_ids, texts)
                texts, node_ids = [], []

        # 处理剩余的节点
        if texts:
            self.process_batch(node_ids, texts)

        # 保存缓存
        self._save_cache()
        self.logger.info(f"'{node_label}' 节点向量生成完成")

    def generate_all_embeddings(self, force: bool = False):
        """
        为配置文件中所有节点类型生成向量嵌入

        Args:
            force: 是否强制重新生成（忽略缓存）
        """
        self.logger.info("=" * 80)
        self.logger.info("开始为所有节点类型生成向量嵌入")
        self.logger.info("=" * 80)

        node_types_config = self.config.get("node_types", {})

        # 按priority排序
        sorted_node_types = sorted(
            node_types_config.items(),
            key=lambda x: x[1].get("priority", 999)
        )

        for node_label, config in sorted_node_types:
            vec_keys = config.get("vec_keys", [])

            if not vec_keys:
                self.logger.warning(f"节点类型 '{node_label}' 未配置vec_keys，跳过")
                continue

            try:
                self.generate_embeddings_for_node_type(node_label, vec_keys, force)
            except Exception as e:
                self.logger.error(f"为节点类型 '{node_label}' 生成向量失败: {e}", exc_info=True)

        self.logger.info("=" * 80)
        self.logger.info("所有节点类型向量嵌入生成完成")
        self.logger.info("=" * 80)
