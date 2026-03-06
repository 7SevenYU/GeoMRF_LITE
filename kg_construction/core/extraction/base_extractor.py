from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any
from kg_construction.core.extraction.id_generator import IDGenerator


@dataclass
class ExtractedNode:
    node_id: str
    node_type: str
    node_label: str
    cypher_label: str
    attributes: Dict[str, Any]
    merge_keys: List[str]
    category: str
    confidence: float = 1.0
    extraction_method: str = ""


@dataclass
class ExtractedRelation:
    relation_id: str
    relation_type: str
    relation_label: str
    cypher_label: str
    head_node_id: str
    tail_node_id: str
    head_merge_key: str
    tail_merge_key: str
    confidence: float = 1.0
    extraction_method: str = ""


@dataclass
class ExtractionResult:
    chunk_id: str
    document_type: str
    source_file: str
    nodes: List[ExtractedNode] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    def __init__(self, config: Dict[str, Any], id_generator: IDGenerator):
        self.config = config
        self.id_gen = id_generator

    @abstractmethod
    def extract(self, text: str, **kwargs) -> ExtractionResult:
        pass
