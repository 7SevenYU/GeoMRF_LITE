import json
from pathlib import Path
from typing import Optional
from kg_construction.utils.logger import setup_logger


def _get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent.parent


class IDGenerator:
    """全局ID生成器，支持持久化"""

    def __init__(self, state_dir: Optional[str] = None):
        if state_dir is None:
            project_root = _get_project_root()
            state_dir = project_root / "kg_construction" / "data" / "processed" / "extraction_state"
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "id_counter.json"
        self.logger = setup_logger("IDGenerator", "extraction.log")

        # 初始化计数器
        self.chunk_counter = 0
        self.node_counter = 0
        self.relation_counter = 0

        # 加载已有计数器
        self._load_counters()

    def _load_counters(self):
        """从文件加载计数器"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    counters = json.load(f)
                    self.chunk_counter = counters.get("chunk_counter", 0)
                    self.node_counter = counters.get("node_counter", 0)
                    self.relation_counter = counters.get("relation_counter", 0)
                    self.logger.info(f"加载已有计数器: chunk={self.chunk_counter}, node={self.node_counter}, relation={self.relation_counter}")
            else:
                self.logger.info("未找到计数器文件，从0开始")
        except Exception as e:
            self.logger.warning(f"加载计数器失败: {e}，从0开始")

    def _save_counters(self):
        """保存计数器到文件"""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            counters = {
                "chunk_counter": self.chunk_counter,
                "node_counter": self.node_counter,
                "relation_counter": self.relation_counter
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(counters, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"保存计数器失败: {e}")

    def generate_chunk_id(self) -> str:
        """
        生成全局唯一chunk ID

        Returns:
            str: 格式为 chunk_XXXXXX 的ID
        """
        self.chunk_counter += 1
        chunk_id = f"chunk_{self.chunk_counter:06d}"
        self._save_counters()
        return chunk_id

    def generate_node_id(self) -> str:
        """
        生成全局唯一节点ID

        Returns:
            str: 格式为 node_XXXXXX 的ID
        """
        self.node_counter += 1
        node_id = f"node_{self.node_counter:06d}"
        self._save_counters()
        return node_id

    def generate_relation_id(self) -> str:
        """
        生成全局唯一关系ID

        Returns:
            str: 格式为 rel_XXXXXX 的ID
        """
        self.relation_counter += 1
        relation_id = f"rel_{self.relation_counter:06d}"
        self._save_counters()
        return relation_id
