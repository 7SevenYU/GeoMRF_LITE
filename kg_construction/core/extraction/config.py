import sys
from enum import Enum
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ExtractorType(Enum):
    REGEX = "regex"
    SEMANTIC = "semantic"
    JSON = "json"


class SemanticExtractorType(Enum):
    LEXICON = "lexicon"
    LLM = "llm"


class NodeCategory(Enum):
    S = "S"
    G = "G"
    C = "C"
    P = "P"


REGEX_PATTERNS = {
    "chainage": r'(?:D?K)?\d+\+\d+(?:\.\d+)?',
    "chainage_range": r'(?:D?K)?\d+\+\d+(?:\.\d+)?\s*[~-~]\s*(?:D?K)?\d+\+\d+(?:\.\d+)?',
    "time": r'\d{4}[-年]\d{1,2}[-月]\d{1,2}[日号]?',
    "datetime": r'\d{4}[-年]\d{1,2}[-月]\d{1,2}[日号]?\s*\d{1,2}:\d{2}'
}


# 字典文件路径配置（在extraction模块内部）
EXTRACTION_DIR = Path(__file__).parent
LEXICON_DIR = EXTRACTION_DIR / "data" / "lexicons"

LEXICON_CONFIG = {
    "geo_condition": {
        "root_csv": str(LEXICON_DIR / "root_match_lexicon_zh_v3_4454_no_attr_overlap.csv"),
        "attr_csv": str(LEXICON_DIR / "attribute_lexicon_step2_zh_v2_556_no_root_overlap.csv"),
        "extractor_module": "lexicons.by_lexicons",
        "extractor_class": "GeoConditionExtractor"
    },
    "risk_type": {
        "keywords": ["岩爆", "掉块", "突涌", "塌方", "富水破碎带"]
    },
    "rock_grade": {
        "keywords": ["II级", "III级", "IV级", "V级", "2级", "3级", "4级", "5级", "Ⅱ级", "Ⅲ级", "Ⅳ级", "Ⅴ级"]
    },
    "risk_level": {
        "keywords": ["Low", "Medium", "High", "低", "中", "高"]
    }
}


NODE_TYPES = {
    "变更信息": {
        "label": "CHANGE_INFORMATION",
        "category": NodeCategory.S,
        "cypher_label": "变更信息",
        "attributes": ["chainage", "information"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "施工信息": {
        "label": "CONSTRUCTION_INFORMATION",
        "category": NodeCategory.S,
        "cypher_label": "施工信息",
        "attributes": ["chainage", "information"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "时间": {
        "label": "TIME",
        "category": NodeCategory.S,
        "cypher_label": "时间",
        "attributes": ["time"],
        "default_extractor": ExtractorType.REGEX
    },
    "设计信息": {
        "label": "DESIGN_INFORMATION",
        "category": NodeCategory.S,
        "cypher_label": "设计信息",
        "attributes": ["chainage", "information"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "探测方法": {
        "label": "DETECTION_METHOD",
        "category": NodeCategory.G,
        "cypher_label": "探测方法",
        "attributes": ["detectionMethod"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "探测结论": {
        "label": "DETECTION_CONCLUSION",
        "category": NodeCategory.G,
        "cypher_label": "探测结论",
        "attributes": ["detectionConclusion", "geologicalElements", "后续建议"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "围岩等级": {
        "label": "SURROUNDING_ROCK_GRADE",
        "category": NodeCategory.G,
        "cypher_label": "围岩等级",
        "attributes": ["grade"],
        "default_extractor": ExtractorType.SEMANTIC,
        "normalization": "remove_suffix"
    },
    "风险类型": {
        "label": "RISK_TYPE",
        "category": NodeCategory.C,
        "cypher_label": "风险类型",
        "attributes": ["riskType"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "预警等级": {
        "label": "WARNING_GRADE",
        "category": NodeCategory.C,
        "cypher_label": "预警等级",
        "attributes": ["warningGrade"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "地质风险等级": {
        "label": "GEOLOGICAL_RISK_GRADE",
        "category": NodeCategory.C,
        "cypher_label": "地质风险等级",
        "attributes": ["geologicalRiskGrade"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "风险评估": {
        "label": "RISK_ASSESSMENT",
        "category": NodeCategory.C,
        "cypher_label": "风险评估",
        "attributes": ["riskAssessment"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "施工规范": {
        "label": "CONSTRUCTION_SPECIFICATIONS",
        "category": NodeCategory.P,
        "cypher_label": "施工规范",
        "attributes": ["constructionSpecifications", "applicableConditions"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "紧急响应措施": {
        "label": "EMERGENCY_RESPONSE_GUIDELINES",
        "category": NodeCategory.P,
        "cypher_label": "紧急响应措施",
        "attributes": ["emergencyResponseGuidelines", "applicableConditions"],
        "default_extractor": ExtractorType.SEMANTIC
    },
    "历史处置案例": {
        "label": "HISTORICAL_MITIGATION_CASE",
        "category": NodeCategory.P,
        "cypher_label": "历史处置案例",
        "attributes": ["s_id", "warningDate", "riskDescription", "chainage"],
        "default_extractor": ExtractorType.SEMANTIC
    }
}


RELATION_TYPES = {
    "HAS_SPATIOTEMPORAL": {
        "label": "HAS_SPATIOTEMPORAL",
        "cypher_label": "HAS_SPATIOTEMPORAL"
    },
    "IS_ASSOCIATED_WITH": {
        "label": "IS_ASSOCIATED_WITH",
        "cypher_label": "IS_ASSOCIATED_WITH"
    },
    "HAS_RISK_ASSESSMENT": {
        "label": "HAS_RISK_ASSESSMENT",
        "cypher_label": "HAS_RISK_ASSESSMENT"
    },
    "OCCURS_AT": {
        "label": "OCCURS_AT",
        "cypher_label": "OCCURS_AT"
    },
    "WAS_SURVEYED_BY": {
        "label": "WAS_SURVEYED_BY",
        "cypher_label": "WAS_SURVEYED_BY"
    },
    "INDICATES": {
        "label": "INDICATES",
        "cypher_label": "INDICATES"
    },
    "HAS_SURROUNDING_ROCK_GRADE": {
        "label": "HAS_SURROUNDING_ROCK_GRADE",
        "cypher_label": "HAS_SURROUNDING_ROCK_GRADE"
    },
    "HAS_WARNING_GRADE": {
        "label": "HAS_WARNING_GRADE",
        "cypher_label": "HAS_WARNING_GRADE"
    },
    "HAS_RISK_TYPE": {
        "label": "HAS_RISK_TYPE",
        "cypher_label": "HAS_RISK_TYPE"
    },
    "REFERS_TO": {
        "label": "REFERS_TO",
        "cypher_label": "REFERS_TO"
    },
    "RESPONDS_TO": {
        "label": "RESPONDS_TO",
        "cypher_label": "RESPONDS_TO"
    },
    "CONSIDERS": {
        "label": "CONSIDERS",
        "cypher_label": "CONSIDERS"
    }
}


RELATION_INFERENCE_CONFIG = {
    "chainage_match": {
        "IS_ASSOCIATED_WITH": {
            "source_nodes": ["变更信息", "施工信息"],
            "target_node": "设计信息",
            "match_type": "range_overlap",
            "description": "通过里程区间匹配建立关联关系"
        },
        "OCCURS_AT": {
            "source_node": "历史处置案例",
            "target_node": "施工信息",
            "match_type": "range_overlap",
            "description": "历史处置案例通过里程匹配到施工信息"
        }
    },

    "multi_hop": {
        "WAS_SURVEYED_BY": {
            "source_node": "历史处置案例",
            "target_node": "探测方法",
            "inference_path": [
                ("历史处置案例", "OCCURS_AT", "施工信息"),
                ("施工信息", "WAS_SURVEYED_BY", "探测方法")
            ],
            "description": "历史处置案例通过施工信息关联到探测方法"
        },

        "considers_geological_risk": {
            "source_node": "紧急响应措施",
            "target_node": "地质风险等级",
            "inference_path": [
                ("紧急响应措施", "RESPONDS_TO", "历史处置案例"),
                ("历史处置案例", "OCCURS_AT", "施工信息"),
                ("施工信息", "WAS_SURVEYED_BY", "探测方法"),
                ("探测方法", "INDICATES", "地质风险等级")
            ],
            "description": "紧急响应措施通过多跳路径关联到地质风险等级"
        },

        "considers_detection_conclusion": {
            "source_node": "紧急响应措施",
            "target_node": "探测结论",
            "inference_path": [
                ("紧急响应措施", "RESPONDS_TO", "历史处置案例"),
                ("历史处置案例", "OCCURS_AT", "施工信息"),
                ("施工信息", "WAS_SURVEYED_BY", "探测方法"),
                ("探测方法", "INDICATES", "探测结论")
            ],
            "description": "紧急响应措施通过多跳路径关联到探测结论"
        },

        "considers_rock_grade": {
            "source_node": "紧急响应措施",
            "target_node": "围岩等级",
            "inference_path": [
                ("紧急响应措施", "RESPONDS_TO", "历史处置案例"),
                ("历史处置案例", "OCCURS_AT", "施工信息"),
                ("施工信息", "WAS_SURVEYED_BY", "探测方法"),
                ("探测方法", "INDICATES", "围岩等级")
            ],
            "description": "紧急响应措施通过多跳路径关联到围岩等级"
        },

        "considers_warning_grade": {
            "source_node": "紧急响应措施",
            "target_node": "预警等级",
            "inference_path": [
                ("紧急响应措施", "RESPONDS_TO", "历史处置案例"),
                ("历史处置案例", "HAS_WARNING_GRADE", "预警等级")
            ],
            "description": "紧急响应措施通过历史处置案例关联到预警等级"
        },

        "considers_risk_assessment_with_chainage": {
            "source_node": "紧急响应措施",
            "target_node": "风险评估",
            "condition": "has_chainage",
            "inference_path": [
                ("紧急响应措施", "chainage_match", "设计信息"),
                ("设计信息", "HAS_RISK_ASSESSMENT", "风险评估")
            ],
            "description": "有里程的紧急响应措施直接匹配设计信息"
        },

        "considers_risk_assessment_without_chainage": {
            "source_node": "紧急响应措施",
            "target_node": "风险评估",
            "condition": "no_chainage",
            "inference_path": [
                ("紧急响应措施", "RESPONDS_TO", "历史处置案例"),
                ("历史处置案例", "OCCURS_AT", "施工信息"),
                ("施工信息", "IS_ASSOCIATED_WITH", "设计信息"),
                ("设计信息", "HAS_RISK_ASSESSMENT", "风险评估")
            ],
            "description": "无里程的紧急响应措施通过多跳路径关联到风险评估"
        }
    },

    "manual_mapping": {
        "REFERS_TO": {
            "source_node": "紧急响应措施",
            "target_node": "施工规范",
            "mapping_rules": {
                "default": []
            },
            "description": "紧急响应措施到施工规范的映射，通过人工定义的规则或语义匹配"
        }
    }
}


DOCUMENT_EXTRACTION_CONFIG = {
    "变更纪要": {
        "source_format": "pdf",
        "semantic_extractor": SemanticExtractorType.LLM,

        "nodes": {
            "变更信息": {
                "required": True,
                "attributes": ["chainage", "information"],
                "description": "变更信息节点"
            },
            "时间": {
                "required": True,
                "attributes": ["time"],
                "description": "时间节点"
            },
            "风险类型": {
                "required": True,
                "attributes": ["riskType"],
                "description": "风险类型节点"
            },
            "围岩等级": {
                "required": False,
                "attributes": ["grade"],
                "description": "围岩等级节点"
            },
            "紧急响应措施": {
                "required": True,
                "attributes": ["emergencyResponseGuidelines", "applicableConditions", "s_id", "riskType", "keywords"],
                "description": "紧急响应措施节点"
            },
            "历史处置案例": {
                "required": True,
                "attributes": ["warningDate", "riskDescription", "s_id", "chainage"],
                "description": "历史处置案例节点"
            }
        },

        "relations": {
            "HAS_SPATIOTEMPORAL": {
                "head": "变更信息",
                "tail": "时间",
                "required": True
            },
            "HAS_RISK_TYPE": {
                "head": "历史处置案例",
                "tail": "风险类型",
                "required": True
            },
            "HAS_SURROUNDING_ROCK_GRADE": {
                "head": "历史处置案例",
                "tail": "围岩等级",
                "required": False
            },
            "considers_emergency_rock": {
                "relation": "CONSIDERS",
                "head": "紧急响应措施",
                "tail": "围岩等级",
                "required": False
            },
            "respondsTo_emergency": {
                "relation": "RESPONDS_TO",
                "head": "紧急响应措施",
                "tail": "风险类型",
                "required": True
            }
        }
    },

    "超前地质预报_水平声波剖面": {
        "source_format": "json",

        "nodes": {
            "施工信息": {
                "required": True,
                "attributes": ["chainage", "information"],
                "json_field_mapping": {
                    "chainage": "里程范围",
                    "information": "__ALL_REMAINING__"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "施工信息节点"
            },
            "时间": {
                "required": True,
                "attributes": ["time"],
                "json_field_mapping": {
                    "time": "上传日期"
                },
                "description": "时间节点"
            },
            "探测方法": {
                "required": True,
                "attributes": ["detectionMethod", "chainage"],
                "metadata_field": "detection_method",
                "json_field_mapping": {
                    "chainage": "里程范围"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "探测方法节点（每段独立）"
            },
            "探测结论": {
                "required": True,
                "attributes": ["detectionConclusion", "geologicalElements"],
                "json_field_mapping": {
                    "detectionConclusion": "探测结论"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "探测结论节点"
            },
            "地质风险等级": {
                "required": False,
                "attributes": ["geologicalRiskGrade", "chainage"],
                "json_field_mapping": {
                    "geologicalRiskGrade": "地质风险等级",
                    "chainage": "里程范围"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "地质风险等级节点"
            }
        },

        "relations": {
            "HAS_SPATIOTEMPORAL": {
                "head": "施工信息",
                "tail": "时间",
                "required": True,
                "json_defined": True
            },
            "WAS_SURVEYED_BY": {
                "head": "施工信息",
                "tail": "探测方法",
                "required": True,
                "json_defined": True
            },
            "indicates_detection_conclusion": {
                "relation": "INDICATES",
                "head": "探测方法",
                "tail": "探测结论",
                "required": True,
                "json_defined": True
            },
            "indicates_risk_grade": {
                "relation": "INDICATES",
                "head": "探测方法",
                "tail": "地质风险等级",
                "required": True,
                "json_defined": True
            }
        }
    },

    "超前地质预报_TSP": {
        "source_format": "json",

        "nodes": {
            "施工信息": {
                "required": True,
                "attributes": ["chainage", "information"],
                "json_field_mapping": {
                    "chainage": "里程范围",
                    "information": "__ALL_REMAINING__"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "施工信息节点"
            },
            "时间": {
                "required": True,
                "attributes": ["time"],
                "json_field_mapping": {
                    "time": "上传日期"
                },
                "description": "时间节点"
            },
            "探测方法": {
                "required": True,
                "attributes": ["detectionMethod", "chainage"],
                "metadata_field": "detection_method",
                "json_field_mapping": {
                    "chainage": "里程范围"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "探测方法节点（每段独立）"
            },
            "探测结论": {
                "required": True,
                "attributes": ["detectionConclusion", "geologicalElements"],
                "json_field_mapping": {
                    "detectionConclusion": "探测结论"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "探测结论节点"
            },
            "地质风险等级": {
                "required": False,
                "attributes": ["geologicalRiskGrade", "chainage"],
                "json_field_mapping": {
                    "geologicalRiskGrade": "地质风险等级",
                    "chainage": "里程范围"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "地质风险等级节点"
            }
        },

        "relations": {
            "HAS_SPATIOTEMPORAL": {
                "head": "施工信息",
                "tail": "时间",
                "required": True,
                "json_defined": True
            },
            "WAS_SURVEYED_BY": {
                "head": "施工信息",
                "tail": "探测方法",
                "required": True,
                "json_defined": True
            },
            "indicates_detection_conclusion": {
                "relation": "INDICATES",
                "head": "探测方法",
                "tail": "探测结论",
                "required": True,
                "json_defined": True
            },
            "indicates_risk_grade": {
                "relation": "INDICATES",
                "head": "探测方法",
                "tail": "地质风险等级",
                "required": True,
                "json_defined": True
            }
        }
    },

    "超前地质预报_地质雷达": {
        "source_format": "json",

        "nodes": {
            "施工信息": {
                "required": True,
                "attributes": ["chainage", "information"],
                "json_field_mapping": {
                    "chainage": "里程范围",
                    "information": "__ALL_REMAINING__"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "施工信息节点"
            },
            "时间": {
                "required": True,
                "attributes": ["time"],
                "json_field_mapping": {
                    "time": "上传日期"
                },
                "description": "时间节点"
            },
            "探测方法": {
                "required": True,
                "attributes": ["detectionMethod", "chainage"],
                "metadata_field": "detection_method",
                "json_field_mapping": {
                    "chainage": "里程范围"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "探测方法节点（每段独立）"
            },
            "探测结论": {
                "required": True,
                "attributes": ["detectionConclusion", "geologicalElements"],
                "json_field_mapping": {
                    "detectionConclusion": "探测结论"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "探测结论节点"
            },
            "地质风险等级": {
                "required": False,
                "attributes": ["geologicalRiskGrade", "chainage"],
                "json_field_mapping": {
                    "geologicalRiskGrade": "地质风险等级",
                    "chainage": "里程范围"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "地质风险等级节点"
            }
        },

        "relations": {
            "HAS_SPATIOTEMPORAL": {
                "head": "施工信息",
                "tail": "时间",
                "required": True,
                "json_defined": True
            },
            "WAS_SURVEYED_BY": {
                "head": "施工信息",
                "tail": "探测方法",
                "required": True,
                "json_defined": True
            },
            "indicates_detection_conclusion": {
                "relation": "INDICATES",
                "head": "探测方法",
                "tail": "探测结论",
                "required": True,
                "json_defined": True
            },
            "indicates_risk_grade": {
                "relation": "INDICATES",
                "head": "探测方法",
                "tail": "地质风险等级",
                "required": True,
                "json_defined": True
            }
        }
    },

    "超前地质预报_洞身纵向地质素描": {
        "source_format": "json",

        "nodes": {
            "施工信息": {
                "required": True,
                "attributes": ["chainage", "information"],
                "json_field_mapping": {
                    "chainage": "里程",
                    "information": "__ALL_REMAINING__"
                },
                "description": "施工信息节点"
            },
            "探测方法": {
                "required": True,
                "attributes": ["detectionMethod", "chainage"],
                "metadata_field": "detection_method",
                "json_field_mapping": {
                    "chainage": "里程"
                },
                "is_array": True,
                "array_field": "结论",
                "description": "探测方法节点（每段独立）"
            },
            "围岩等级": {
                "required": True,
                "attributes": ["grade"],
                "json_field_mapping": {
                    "grade": "施工围岩分级"
                },
                "description": "围岩等级节点"
            },
            "探测结论": {
                "required": True,
                "attributes": ["detectionConclusion", "后续建议"],
                "json_field_mapping": {
                    "detectionConclusion": "结论.探测结论",
                    "后续建议": "后续建议"
                },
                "description": "探测结论节点"
            },
            "地质风险等级": {
                "required": True,
                "attributes": ["geologicalRiskGrade"],
                "json_field_mapping": {
                    "geologicalRiskGrade": "结论.地质风险等级"
                },
                "description": "地质风险等级节点"
            },
            "风险类型": {
                "required": True,
                "attributes": ["riskType"],
                "json_field_mapping": {
                    "riskType": "结论.风险类别"
                },
                "description": "风险类型节点"
            }
        },

        "relations": {
            "WAS_SURVEYED_BY": {
                "head": "施工信息",
                "tail": "探测方法",
                "required": True,
                "json_defined": True
            },
            "HAS_SURROUNDING_ROCK_GRADE": {
                "head": "施工信息",
                "tail": "围岩等级",
                "required": True,
                "json_defined": True
            },
            "INDICATES": {
                "head": "探测方法",
                "tail": "探测结论",
                "required": True,
                "json_defined": True
            },
            "indicates_grade": {
                "relation": "INDICATES",
                "head": "探测方法",
                "tail": "地质风险等级",
                "required": True,
                "json_defined": True
            },
            "HAS_RISK_TYPE": {
                "relation": "HAS_RISK_TYPE",
                "head": "施工信息",
                "tail": "风险类型",
                "required": True,
                "json_defined": True
            }
        }
    },

    "处置记录": {
        "source_format": "json",

        "nodes": {
            "历史处置案例": {
                "required": True,
                "attributes": ["s_id", "warningDate", "riskDescription", "chainage"],
                "json_field_mapping": {
                    "s_id": "方案id",
                    "warningDate": "预警时间",
                    "riskDescription": "风险描述",
                    "chainage": "分段位置"
                },
                "use_lexicon_for": [],
                "description": "历史处置案例节点"
            },
            "风险类型": {
                "required": True,
                "attributes": ["riskType"],
                "json_field_mapping": {
                    "riskType": "风险类型"
                },
                "description": "风险类型节点"
            },
            "预警等级": {
                "required": True,
                "attributes": ["warningGrade"],
                "json_field_mapping": {
                    "warningGrade": "预警等级"
                },
                "description": "预警等级节点"
            },
            "围岩等级": {
                "required": True,
                "attributes": ["grade"],
                "json_field_mapping": {
                    "grade": "围岩等级"
                },
                "description": "围岩等级节点"
            },
            "探测方法": {
                "required": False,
                "attributes": ["detectionMethod", "chainage"],
                "json_field_mapping": {
                    "detectionMethod": "探测方法"
                },
                "description": "探测方法节点（可选，包含chainage）"
            }
        },

        "relations": {
            "HAS_SURROUNDING_ROCK_GRADE": {
                "head": "历史处置案例",
                "tail": "围岩等级",
                "required": True,
                "json_defined": True,
                "description": "历史处置案例关联围岩等级"
            },
            "HAS_WARNING_GRADE": {
                "head": "历史处置案例",
                "tail": "预警等级",
                "required": True,
                "json_defined": True,
                "description": "历史处置案例关联预警等级"
            },
            "HAS_RISK_TYPE": {
                "head": "历史处置案例",
                "tail": "风险类型",
                "required": True,
                "json_defined": True,
                "description": "历史处置案例关联风险类型"
            },
            "WAS_SURVEYED_BY": {
                "head": "历史处置案例",
                "tail": "探测方法",
                "required": False,
                "json_defined": True,
                "description": "历史处置案例关联探测方法（可选）"
            }
        }
    },

    "设计信息(PDF)": {
        "source_format": "pdf",
        "semantic_extractor": SemanticExtractorType.LLM,

        "nodes": {
            "施工规范": {
                "required": True,
                "attributes": ["constructionSpecifications", "riskType"],
                "description": "施工规范节点"
            },
            "风险类型": {
                "required": True,
                "attributes": ["riskType"],
                "description": "风险类型节点"
            }
        },

        "relations": {
            "RESPONDS_TO": {
                "head": "施工规范",
                "tail": "风险类型",
                "required": True,
                "description": "施工规范响应风险类型"
            }
        }
    },

    "设计信息(JSON)": {
        "source_format": "json",

        "nodes": {
            "设计信息": {
                "required": True,
                "attributes": ["序号", "chainage", "length", "information", "grade"],
                "json_field_mapping": {
                    "序号": "序号",
                    "length": "长度",
                    "information": "衬砌名称",
                    "grade": "围岩等级"
                },
                "composite_fields": {
                    "chainage": {
                        "fields": ["起始里程", "终止里程"],
                        "separator": "～"
                    }
                },
                "description": "设计信息节点"
            },
            "风险评估": {
                "required": True,
                "attributes": ["chainage", "riskAssessments"],
                "metadata_field": "parent_chainage",
                "json_field_mapping": {
                    "chainage": "parent_chainage",
                    "riskAssessments": "风险评估"
                },
                "array_to_string": True,
                "description": "风险评估节点（包含所有风险类型的数组）"
            }
        },

        "relations": {
            "has_risk_assessment": {
                "relation": "HAS_RISK_ASSESSMENT",
                "head": "设计信息",
                "tail": "风险评估",
                "json_defined": True,
                "description": "设计信息包含风险评估"
            }
        }
    },

    "专家经验": {
        "source_format": "json",

        "nodes": {
            "紧急响应措施": {
                "required": True,
                "attributes": ["emergencyResponseGuidelines", "applicableConditions", "s_id", "riskType", "keywords"],
                "json_field_mapping": {
                    "emergencyResponseGuidelines": "处置方案",
                    "applicableConditions": "适用条件",
                    "s_id": "s_id",
                    "riskType": "风险类型",
                    "keywords": "关键词"
                },
                "description": "紧急响应措施节点（专家经验）"
            },
            "风险类型": {
                "required": True,
                "attributes": ["riskType"],
                "json_field_mapping": {
                    "riskType": "风险类型"
                },
                "description": "风险类型节点"
            }
        },

        "relations": {
            "RESPONDS_TO": {
                "relation": "RESPONDS_TO",
                "head": "紧急响应措施",
                "tail": "风险类型",
                "required": True,
                "json_defined": True,
                "description": "紧急响应措施响应风险类型"
            }
        }
    }
}


DOCUMENT_AGGREGATION_CONFIG = {
    "变更纪要": {
        "aggregation_level": "document",
        "node_strategies": {
            "变更信息": "merge_attributes",
            "时间": "select_earliest",
            "风险类型": "select_most_common",
            "围岩等级": "select_most_common",
            "紧急响应措施": "merge_descriptions",
            "历史处置案例": "create_single"
        },
        "description": "变更纪要采用文档级聚合，智能选择最频繁出现的信息作为主要信息"
    },
    "超前地质预报_水平声波剖面": {
        "aggregation_level": "chunk",
        "description": "超前地质预报保持chunk级抽取，每条探测记录独立"
    },
    "超前地质预报_TSP": {
        "aggregation_level": "chunk",
        "description": "TSP探测保持chunk级抽取"
    },
    "超前地质预报_地质雷达": {
        "aggregation_level": "chunk",
        "description": "地质雷达探测保持chunk级抽取"
    },
    "处置记录": {
        "aggregation_level": "chunk",
        "description": "处置记录保持chunk级抽取"
    },
    "专家经验": {
        "aggregation_level": "chunk",
        "description": "专家经验保持chunk级抽取"
    },
    "设计信息(JSON)": {
        "aggregation_level": "chunk",
        "description": "设计信息保持chunk级抽取"
    },
    "超前地质预报_洞身纵向地质素描": {
        "aggregation_level": "chunk",
        "description": "洞身纵向地质素描保持chunk级抽取"
    }
}


# ====================================================================
# 节点属性标准化Schema
# 确保同一类型的节点在Neo4j中有相同的属性集合
# ====================================================================

NODE_ATTRIBUTE_SCHEMAS = {
    "紧急响应措施": [
        "emergencyResponseGuidelines",
        "applicableConditions",
        "s_id",
        "riskType",
        "keywords"
    ],
    "历史处置案例": [
        "s_id",
        "warningDate",
        "riskDescription",
        "chainage"
    ],
    "变更信息": [
        "chainage",
        "information"
    ],
    "时间": [
        "time"
    ],
    "风险类型": [
        "riskType"
    ],
    "围岩等级": [
        "grade"
    ],
    "预警等级": [
        "warningGrade"
    ],
    "地质风险等级": [
        "geologicalRiskGrade"
    ],
    "施工信息": [
        "chainage"
    ],
    "设计信息": [
        "chainage",
        "information"
    ],
    "探测方法": [
        "detectionMethod",
        "chainage"
    ],
    "探测结论": [
        "detectionConclusion",
        "geologicalElements",
        "后续建议"
    ],
    "施工规范": [
        "constructionSpecifications",
        "riskType"
    ]
}
