import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class Neo4jSyncTracker:
    def __init__(self):
        self.module_dir = Path(__file__).parent
        self.mapping_file = self.module_dir / "neo4j_sync_status.json"
        self.sync_records = self._load_records()

    def _load_records(self) -> Dict:
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                return {}
        return {}

    def is_synced(self, source_file: str) -> bool:
        return source_file in self.sync_records

    def get_sync_info(self, source_file: str) -> Optional[Dict]:
        return self.sync_records.get(source_file)

    def mark_synced(
        self,
        source_file: str,
        node_count: int,
        relation_count: int
    ):
        self.sync_records[source_file] = {
            "status": "completed",
            "sync_time": datetime.now().isoformat(),
            "node_count": node_count,
            "relation_count": relation_count
        }
        self._save_records()

    def reset_file(self, source_file: str):
        if source_file in self.sync_records:
            del self.sync_records[source_file]
            self._save_records()

    def get_all_records(self) -> Dict:
        return self.sync_records.copy()

    def _save_records(self):
        self.module_dir.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.sync_records, f, ensure_ascii=False, indent=2)

    def reset_all(self):
        self.sync_records = {}
        self._save_records()
