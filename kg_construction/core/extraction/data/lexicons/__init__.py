"""
地质条件字典提取模块

包含基于字典的地质实体提取功能
"""

from .by_lexicons import (
    GeoConditionExtractor,
    RootLexicon,
    AttrLexicon,
    ExtractConfig,
)

__all__ = [
    "GeoConditionExtractor",
    "RootLexicon",
    "AttrLexicon",
    "ExtractConfig",
]
