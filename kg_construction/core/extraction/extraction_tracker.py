import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class ExtractionTracker:
    """跟踪文档提取状态，避免重复处理"""

    def __init__(self):
        self.module_dir = Path(__file__).parent
        self.mapping_file = self.module_dir / "extraction_mapping.json"
        self.extraction_records = self._load_records()

    def _load_records(self) -> Dict:
        """加载提取记录"""
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                return {}
        return {}

    def is_extracted(self, source_file: str) -> bool:
        """检查文档是否已提取"""
        return source_file in self.extraction_records

    def get_extraction_info(self, source_file: str) -> Optional[Dict]:
        """获取提取信息"""
        return self.extraction_records.get(source_file)

    def mark_extracted(
        self,
        source_file: str,
        chunk_count: int,
        node_count: int,
        relation_count: int,
        extraction_method: str,
        document_type: str = ""
    ):
        """标记文档为已提取"""
        self.extraction_records[source_file] = {
            "status": "completed",
            "extraction_time": datetime.now().isoformat(),
            "chunk_count": chunk_count,
            "node_count": node_count,
            "relation_count": relation_count,
            "extraction_method": extraction_method,
            "document_type": document_type
        }
        self._save_records()

    def reset_file(self, source_file: str):
        """清除提取记录（强制重新提取）"""
        if source_file in self.extraction_records:
            del self.extraction_records[source_file]
            self._save_records()

    def get_all_records(self) -> Dict:
        """获取所有提取记录"""
        return self.extraction_records.copy()

    def _save_records(self):
        """保存记录"""
        self.module_dir.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_records, f, ensure_ascii=False, indent=2)
