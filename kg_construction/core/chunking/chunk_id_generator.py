import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from kg_construction.core.extraction.id_generator import IDGenerator


class ChunkIDGenerator:
    def __init__(self, processed_data_dir: str):
        self.processed_data_dir = Path(processed_data_dir)
        # 获取chunking模块的目录
        self.module_dir = Path(__file__).parent
        self.mapping_file = self.module_dir / "chunk_mapping.json"

        # 自动迁移旧位置的数据
        self._migrate_old_mapping()

        self.processed_files = self._load_processed_files()

        # 使用统一的IDGenerator来生成chunk_id
        self.id_gen = IDGenerator()

    def _migrate_old_mapping(self):
        """从旧位置迁移chunk_mapping.json到新位置"""
        if self.mapping_file.exists():
            return

        old_mapping_file_1 = self.processed_data_dir / "chunk_mapping.json"
        old_mapping_file_2 = self.processed_data_dir / "chunks" / ".chunk_mapping.json"

        # 迁移旧位置1的数据
        if old_mapping_file_1.exists():
            try:
                shutil.copy(old_mapping_file_1, self.mapping_file)
                print(f"已迁移 chunk_mapping.json: {old_mapping_file_1} -> {self.mapping_file}")
            except Exception as e:
                print(f"迁移失败: {e}")

        # 迁移旧位置2的数据
        elif old_mapping_file_2.exists():
            try:
                shutil.copy(old_mapping_file_2, self.mapping_file)
                print(f"已迁移 chunk_mapping.json: {old_mapping_file_2} -> {self.mapping_file}")
            except Exception as e:
                print(f"迁移失败: {e}")

    def _load_processed_files(self) -> Dict[str, int]:
        """加载已处理文件映射"""
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                return {}
        return {}

    def is_processed(self, file_path: str) -> bool:
        """检查文件是否已处理"""
        return file_path in self.processed_files

    def get_file_chunk_count(self, file_path: str) -> Optional[int]:
        """获取已处理文件的chunk数量"""
        return self.processed_files.get(file_path)

    def get_next_chunk_id(self) -> str:
        """生成下一个唯一的chunk_id（使用统一的IDGenerator）"""
        return self.id_gen.generate_chunk_id()

    def mark_file_processed(self, file_path: str, chunk_count: int):
        """标记文件为已处理"""
        self.processed_files[file_path] = chunk_count
        self._save_mapping()

    def reset_file(self, file_path: str):
        """清除某个文件的处理记录（强制重新处理）"""
        if file_path in self.processed_files:
            del self.processed_files[file_path]
            self._save_mapping()

    def get_processed_files(self) -> Dict[str, int]:
        """获取所有已处理文件及其chunk数量"""
        return self.processed_files.copy()

    def _save_mapping(self):
        """保存文件映射"""
        self.module_dir.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_files, f, ensure_ascii=False, indent=2)
