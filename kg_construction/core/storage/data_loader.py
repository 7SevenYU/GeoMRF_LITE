import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging


def _get_project_root():
    """获取项目根目录"""
    # 从 data_loader.py 向上4级到项目根目录
    # data_loader.py -> storage -> core -> kg_construction -> 项目根目录
    return Path(__file__).parent.parent.parent.parent


class DataLoader:
    def __init__(self, extraction_results_dir: str, extraction_mapping_file: str):
        project_root = _get_project_root()

        # 将相对路径转换为绝对路径
        if not Path(extraction_results_dir).is_absolute():
            extraction_results_dir = project_root / extraction_results_dir
        if not Path(extraction_mapping_file).is_absolute():
            extraction_mapping_file = project_root / extraction_mapping_file

        self.extraction_results_dir = Path(extraction_results_dir)
        self.extraction_mapping_file = Path(extraction_mapping_file)
        self.logger = logging.getLogger(__name__)

    def load_extraction_mapping(self) -> Dict:
        if not self.extraction_mapping_file.exists():
            self.logger.warning(f"提取映射文件不存在: {self.extraction_mapping_file}")
            return {}

        with open(self.extraction_mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_document_results(self, source_file: str, document_type: str) -> Tuple[List[Dict], List[Dict]]:
        source_filename = Path(source_file).stem

        # 尝试从多个可能的位置加载结果
        possible_paths = [
            self.extraction_results_dir / document_type / f"{source_filename}.json",
            self.extraction_results_dir / document_type / "aggregated" / f"{source_filename}_aggregated.json"
        ]

        result_file = None
        for path in possible_paths:
            if path.exists():
                result_file = path
                break

        if result_file:
            # 找到了聚合结果文件，直接加载
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            nodes = data.get("nodes", [])
            relations = data.get("relations", [])

            self.logger.info(f"从 {result_file.name} 加载了 {len(nodes)} 个节点和 {len(relations)} 个关系")
            return nodes, relations
        else:
            # 没有找到聚合结果文件，尝试加载该文档的所有chunk文件
            chunk_files = list((self.extraction_results_dir / document_type).glob("chunk_*.json"))

            if not chunk_files:
                self.logger.warning(f"未找到提取结果文件: {source_filename}")
                return [], []

            # 过滤出属于该source_file的chunk
            all_nodes = []
            all_relations = []
            loaded_count = 0

            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)

                    # 检查是否属于该source_file
                    if chunk_data.get("source_file") == source_file:
                        nodes = chunk_data.get("nodes", [])
                        relations = chunk_data.get("relations", [])
                        all_nodes.extend(nodes)
                        all_relations.extend(relations)
                        loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"加载chunk文件失败 {chunk_file.name}: {e}")

            if loaded_count > 0:
                self.logger.info(f"从 {loaded_count} 个chunk文件加载了 {len(all_nodes)} 个节点和 {len(all_relations)} 个关系")
                return all_nodes, all_relations
            else:
                self.logger.warning(f"未找到属于 {source_filename} 的chunk文件")
                return [], []

    def load_all_unsynced_results(
        self,
        synced_files: set
    ) -> List[Tuple[str, str, List[Dict], List[Dict]]]:
        extraction_mapping = self.load_extraction_mapping()
        results = []

        for source_file, extract_info in extraction_mapping.items():
            if source_file in synced_files:
                continue

            if extract_info.get("status") != "completed":
                self.logger.warning(f"文档提取未完成，跳过: {Path(source_file).name}")
                continue

            document_type = extract_info.get("document_type")
            if not document_type:
                self.logger.warning(f"缺少document_type信息: {Path(source_file).name}")
                continue

            nodes, relations = self.load_document_results(source_file, document_type)
            if nodes or relations:
                results.append((source_file, document_type, nodes, relations))

        return results

    def load_all_results(self) -> List[Tuple[str, str, List[Dict], List[Dict]]]:
        extraction_mapping = self.load_extraction_mapping()
        results = []

        for source_file, extract_info in extraction_mapping.items():
            if extract_info.get("status") != "completed":
                continue

            document_type = extract_info.get("document_type")
            if not document_type:
                continue

            nodes, relations = self.load_document_results(source_file, document_type)
            results.append((source_file, document_type, nodes, relations))

        return results

    def get_document_types(self) -> List[str]:
        if not self.extraction_results_dir.exists():
            return []

        document_types = []
        for item in self.extraction_results_dir.iterdir():
            if item.is_dir() and item.name not in ["knowledge_graph_data"]:
                document_types.append(item.name)

        return document_types
