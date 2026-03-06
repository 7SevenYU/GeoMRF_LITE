"""
里程解析器：解析和比较铁路里程
"""

import re
from typing import Tuple, Optional


class ChainageParser:
    """里程解析和比较工具类"""

    @staticmethod
    def parse(chainage_str: str) -> Optional[Tuple[float, float, str]]:
        """
        解析里程字符串为标准格式

        Args:
            chainage_str: 里程字符串，格式如：
                - "DyK1012+865.5～DyK1013+247" （区间）
                - "DK13+250.00~DK13+274.00" （区间）
                - "DK13+18" （单点）

        Returns:
            (start_meters, end_meters, prefix) 或 None
            - start_meters: 起始里程（米）
            - end_meters: 终止里程（米），单点时等于start
            - prefix: 里程前缀（DK、DyK等）

        Examples:
            >>> ChainageParser.parse("DyK1012+865.5～DyK1013+247")
            (1012865.5, 1013247, 'DyK')
            >>> ChainageParser.parse("DK13+250.00~DK13+274.00")
            (13250.0, 13274.0, 'DK')
            >>> ChainageParser.parse("DK13+18")
            (13018, 13018, 'DK')
        """
        if not chainage_str:
            return None

        # 标准化分隔符
        normalized = chainage_str.replace('～', '~').replace('—', '~')

        # 判断是区间还是单点
        if '~' in normalized:
            # 区间格式：DK100+100~DK200+200
            parts = normalized.split('~')
            if len(parts) != 2:
                return None

            start_part, end_part = parts
            start_result = ChainageParser._parse_single_chainage(start_part)
            end_result = ChainageParser._parse_single_chainage(end_part)

            if not start_result or not end_result:
                return None

            start_meters, start_prefix = start_result
            end_meters, end_prefix = end_result

            # 不同前缀不能关联
            if start_prefix != end_prefix:
                return None

            return (start_meters, end_meters, start_prefix)
        else:
            # 单点格式：DK13+18
            result = ChainageParser._parse_single_chainage(normalized)
            if result:
                meters, prefix = result
                return (meters, meters, prefix)
            return None

    @staticmethod
    def _parse_single_chainage(chainage_str: str) -> Optional[Tuple[float, str]]:
        """
        解析单个里程点

        Args:
            chainage_str: 如 "DK13+250" 或 "DyK1012+865.5"

        Returns:
            (meters, prefix) 或 None
        """
        # 匹配里程格式：DK/DyK + 数字 + 数字
        # 例如：DK13+250, DyK1012+865.5
        pattern = r'^([A-Za-z]*K)?(\d+)\+(\d+(?:\.\d+)?)$'
        match = re.match(pattern, chainage_str.strip())

        if not match:
            return None

        prefix = match.group(1) if match.group(1) else ''
        major_km = int(match.group(2))
        minor_m = float(match.group(3))

        # 转换为米：DK13+250 = 13000 + 250 = 13250
        total_meters = major_km * 1000 + minor_m

        return (total_meters, prefix)

    @staticmethod
    def overlaps(range1: Tuple[float, float, str], range2: Tuple[float, float, str]) -> bool:
        """
        判断两个里程区间是否重叠

        Args:
            range1: (start1, end1, prefix1)
            range2: (start2, end2, prefix2)

        Returns:
            True if ranges overlap, False otherwise

        Examples:
            >>> ChainageParser.overlaps((13000, 13500, 'DK'), (13200, 13800, 'DK'))
            True
            >>> ChainageParser.overlaps((13000, 13100, 'DK'), (13200, 13300, 'DK'))
            False
        """
        if not range1 or not range2:
            return False

        start1, end1, prefix1 = range1
        start2, end2, prefix2 = range2

        # 不同前缀不关联
        if prefix1 != prefix2:
            return False

        # 判断重叠：not (end1 < start2 or end2 < start1)
        return not (end1 < start2 or end2 < start1)

    @staticmethod
    def contains(chainage_range: Tuple[float, float, str], point_value: float, point_prefix: str = None) -> bool:
        """
        判断里程区间是否包含某个点

        Args:
            chainage_range: (start, end, prefix)
            point_value: 单个里程点的米数
            point_prefix: 点的前缀（如果为None，忽略前缀检查）

        Returns:
            True if point is in range, False otherwise

        Examples:
            >>> ChainageParser.contains((13000, 13500, 'DK'), 13250, 'DK')
            True
            >>> ChainageParser.contains((13000, 13500, 'DK'), 13800, 'DK')
            False
        """
        if not chainage_range:
            return False

        start, end, prefix = chainage_range

        # 如果指定了前缀，检查前缀是否匹配
        if point_prefix is not None and prefix != point_prefix:
            return False

        # 判断包含：start <= point <= end
        return start <= point_value <= end

    @staticmethod
    def contains_range(outer: Tuple[float, float, str], inner: Tuple[float, float, str]) -> bool:
        """
        判断一个里程区间是否完全包含另一个里程区间

        Args:
            outer: (start1, end1, prefix1) - 外层区间
            inner: (start2, end2, prefix2) - 内层区间

        Returns:
            True if outer完全包含inner, False otherwise

        Examples:
            >>> ChainageParser.contains_range((13000, 14000, 'DK'), (13200, 13800, 'DK'))
            True
            >>> ChainageParser.contains_range((13000, 13500, 'DK'), (13200, 13800, 'DK'))
            False
            >>> ChainageParser.contains_range((13000, 13500, 'DK'), (12800, 13200, 'DK'))
            False
        """
        if not outer or not inner:
            return False

        start1, end1, prefix1 = outer
        start2, end2, prefix2 = inner

        # 不同前缀不关联
        if prefix1 != prefix2:
            return False

        # 判断完全包含：start1 <= start2 且 end1 >= end2
        return start1 <= start2 and end1 >= end2
