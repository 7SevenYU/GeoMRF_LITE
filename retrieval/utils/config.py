from py2neo import Graph
from transformers import BertTokenizer, BertModel
from pathlib import Path
import torch
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 懒加载变量（只加载一次）
_graph = None
_bge_model = None
_model = None
_tokenizer = None


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent


def get_model():
    """获取BERT模型（用于回退）"""
    global _model
    if _model is None:
        print("[config] 加载 BERT 模型")
        project_root = get_project_root()
        bert_path = project_root / "models" / "bert-base-chinese"
        _model = BertModel.from_pretrained(str(bert_path))
        _model.to(device)
        _model.eval()
    return _model


def get_tokenizer():
    """获取BERT tokenizer"""
    global _tokenizer
    if _tokenizer is None:
        print("[config] 加载 tokenizer")
        project_root = get_project_root()
        bert_path = project_root / "models" / "bert-base-chinese"
        _tokenizer = BertTokenizer.from_pretrained(str(bert_path))
    return _tokenizer


def get_bge_model():
    """获取BGE模型（优先使用）"""
    global _bge_model
    if _bge_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("[config] 加载 BGE 检索模型")
            project_root = get_project_root()
            bge_path = project_root / "models" / "bge-large-zh-v1.5"
            _bge_model = SentenceTransformer(str(bge_path))
        except ImportError:
            print("[config] sentence-transformers未安装，请先安装: pip install sentence-transformers")
            _bge_model = None
        except Exception as e:
            print(f"[config] BGE模型加载失败: {e}")
            _bge_model = None
    return _bge_model


def get_graph():
    """获取Neo4j图数据库连接"""
    global _graph
    if _graph is None:
        print("[config] 连接图数据库")
        project_root = get_project_root()

        # 从配置文件读取连接信息
        config_path = project_root / "kg_construction" / "core" / "storage" / "neo4j_config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Neo4j配置文件不存在: {config_path}\n"
                f"请创建配置文件或检查路径是否正确"
            )

        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        _graph = Graph(
            config.get("uri", "bolt://localhost:7687"),
            auth=(config.get("username", "neo4j"), config.get("password")),
            name=config.get("database", "neo4j")
        )
    return _graph


# 模型checkpoint路径
NER_CHECKPOINT_DIR = get_project_root() / "retrieval" / "models" / "ner" / "checkpoints"
DYNAMIC_WEIGHT_CHECKPOINT_DIR = get_project_root() / "retrieval" / "models" / "dynamic_weight" / "checkpoints"

# NER配置
NER_BATCH_SIZE = 8
NER_EPOCHS = 1
NER_LR = 1e-3
NER_DATA = get_project_root() / "retrieval" / "models" / "ner" / "ner_data" / "ner_dataset.jsonl"
NER_SAVE_PATH = NER_CHECKPOINT_DIR
LEXICON_DATA = get_project_root() / "data" / "lexicon.txt"

# Dynamic Weight配置
DW_BATCH_SIZE = 8
DW_EPOCHS = 1
DW_LR = 1e-3
LAMBDA_ENTROPY = 0.05
DYNAMIC_DATA_FILE = get_project_root() / "retrieval" / "models" / "dynamic_weight" / "dynamic_weight_data" / "dynamic_weight_data.txt"
DYNAMIC_SAVE_PATH = DYNAMIC_WEIGHT_CHECKPOINT_DIR

# 检索配置
SEARCH_CORE_THRESHOLD = 0.6
SEARCH_TOP_K = 4
SEARCH_ALPHA = 0.5
SEARCH_BETA = 0.5
