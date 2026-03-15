import json
import re
import logging
from typing import Optional, Tuple, Dict, Any, List

from retrieval.utils import config

logger = logging.getLogger(__name__)


def parse_mileage(mileage_str: str) -> Tuple[float, float, str]:
    """
    解析里程字符串，支持格式：
    - 起止格式 "起:DK13+298止:DK13+335"
    - 区间格式 "DyK1014+531.00~DyK1014+555.00" 或 "DyK1014+531.00～DyK1014+555.00"
    - 单点格式 "DK13+710"
    - 复合格式 "DK13+200~DK13+300"（设计信息的chainage字段）

    返回 (start, end, prefix)，单点时 start == end
    抛出 ValueError 当格式不匹配
    """

    def extract_single_mileage_with_prefix(km_str: str) -> Tuple[float, str]:
        # 兼容 DK、DyK、K 等开头，提取数字部分和前缀
        match = re.search(r'([a-zA-Z]*?K)?(\d+)\+([\d.]+)', km_str)
        if not match:
            raise ValueError(f"无法解析的里程格式: {km_str}")
        prefix = match.group(1) if match.group(1) else ''
        km = int(match.group(2))
        m = float(match.group(3))
        return km * 1000 + m, prefix

    mileage_str = mileage_str.strip()

    # 类型1: 起止格式 "起:DK13+298止:DK13+335"
    m = re.match(r'起:([a-zA-Z0-9\+\.\s]+)止:([a-zA-Z0-9\+\.\s]+)', mileage_str)
    if m:
        start, start_prefix = extract_single_mileage_with_prefix(m.group(1).strip())
        end, end_prefix = extract_single_mileage_with_prefix(m.group(2).strip())
        if start_prefix != end_prefix:
            raise ValueError(f"起止里程前缀不一致: {start_prefix} vs {end_prefix}")
        return start, end, start_prefix

    # 类型2: 区间范围 "DyK1014+531.00~DyK1014+555.00" 或 "DyK1014+531.00～DyK1014+555.00"
    if "~" in mileage_str or "～" in mileage_str:
        # 统一分隔符
        normalized = mileage_str.replace("～", "~")
        parts = normalized.split("~")
        if len(parts) == 2:
            start, start_prefix = extract_single_mileage_with_prefix(parts[0].strip())
            end, end_prefix = extract_single_mileage_with_prefix(parts[1].strip())
            if start_prefix != end_prefix:
                raise ValueError(f"区间里程前缀不一致: {start_prefix} vs {end_prefix}")
            return start, end, start_prefix
        else:
            raise ValueError(f"无法解析的区间格式: {mileage_str}")

    # 类型3: 单点 "DK13+710"
    val, prefix = extract_single_mileage_with_prefix(mileage_str)
    return val, val, prefix


def extract_key_spa(text: str) -> Tuple[Optional[str], str]:
    """
    从文本中提取空间位置信息
    优先提取区间（如DK1012+77至DK1012+87），其次单点（如DK1012+77）

    Args:
        text: 输入文本

    Returns:
        (mileage, line_name): 里程和线路名称
    """
    # 1. 优先尝试提取区间（支持"至"、"~"、"～"）
    range_pattern = r'(?:D?K)?\d+\+\d+(?:\.\d+)?(?:[~至至](?:D?K)?\d+\+\d+(?:\.\d+)?)'
    range_match = re.search(range_pattern, text)

    if range_match:
        # 提取到区间
        range_str = range_match.group()
        # 标准化为 ~ 格式（统一处理"至"、"～"等）
        key_spa = range_str.replace('至', '~').replace('～', '~')
        logger.debug(f"提取到区间里程: {key_spa}")
    else:
        # 2. 回退到单点提取
        mileage_pattern = r'(?:D?K)?\d+\+\d+(?:\.\d+)?'
        mileage_match = re.search(mileage_pattern, text)
        key_spa = mileage_match.group() if mileage_match else None
        if key_spa:
            logger.debug(f"提取到单点里程: {key_spa}")

    # 3. 匹配线路名称（左线/右线）
    line_choices = ["左线", "右线"]
    line_name = "左线"  # 默认左线
    for line_choice in line_choices:
        if line_choice in text:
            line_name = line_choice
            break

    return key_spa, line_name


def extract_key_risk(text: str) -> Optional[str]:
    """
    从文本中提取主要风险类型。将部分同义词统一归类为标准风险类型。

    Args:
        text: 输入文本

    Returns:
        标准风险类型名称，如果未找到则返回None
    """
    risk_mapping = {
        "突泥水涌": "突涌",
        "富水破碎带": "富水破碎带",
        "富水段落": "富水破碎带",
        "富水段": "富水破碎带",
        "富水": "富水破碎带",
        "涌水": "突涌",
        "突涌": "突涌",
        "岩爆": "岩爆",
        "掉块": "掉块",
        "塌方": "塌方"
    }
    sorted_keywords = sorted(risk_mapping.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword in text:
            return risk_mapping[keyword]
    return None


def deep_get(d: Dict, keys: List, default: Any = "") -> Any:
    """
    深度获取字典值

    Args:
        d: 字典
        keys: 键列表
        default: 默认值

    Returns:
        字典中的值，如果任何键不存在则返回默认值
    """
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def safe_get(node: Optional[Dict], key: str, default: Any = None) -> Any:
    """
    安全获取节点属性值

    Args:
        node: 节点字典
        key: 属性键
        default: 默认值

    Returns:
        属性值，如果节点为None或键不存在则返回默认值
    """
    return node.get(key, default) if node else default


def extract_node_properties(node: Optional[Dict], keys: List[str]) -> Dict[str, Any]:
    """
    从节点中提取指定属性

    Args:
        node: 节点字典
        keys: 属性键列表

    Returns:
        包含指定属性的字典
    """
    return {k: node.get(k) for k in keys} if node else {}


def parse_node_list(nodes: List[Dict], expected_keys: List[str]) -> Dict[str, Any]:
    """
    解析节点列表，提取指定属性

    Args:
        nodes: 节点列表
        expected_keys: 期望的属性键列表

    Returns:
        合并后的属性字典
    """
    result = {}
    if len(nodes) > 0:
        for node in nodes:
            for k in expected_keys:
                if k in node:
                    result[k] = node.get(k)
    return result


def parse_risk_assessments(risk_assessments_str: str) -> List[Dict[str, Any]]:
    """
    解析风险评估JSON字符串

    Args:
        risk_assessments_str: JSON格式的风险评估字符串

    Returns:
        风险评估对象列表
    """
    try:
        if isinstance(risk_assessments_str, str):
            assessments = json.loads(risk_assessments_str)
            return assessments if isinstance(assessments, list) else [assessments]
        return []
    except Exception as e:
        logger.error(f"解析风险评估失败: {e}")
        return []


def filter_risk_assessment_by_type(risk_assessments_str: str, risk_type: str) -> Optional[Dict[str, Any]]:
    """
    从风险评估数组中筛选指定风险类型的评估

    Args:
        risk_assessments_str: JSON格式的风险评估字符串
        risk_type: 目标风险类型

    Returns:
        匹配的风险评估对象，如果未找到则返回None
    """
    assessments = parse_risk_assessments(risk_assessments_str)
    for assessment in assessments:
        if assessment.get("风险类型") == risk_type or assessment.get("riskType") == risk_type:
            return assessment
    return None


def kg_plan_relevance_retrieval(
    scheme_id: int,
    graph: Optional[object] = None
) -> Dict[str, Any]:
    """
    根据紧急响应措施ID查询关联的历史风险、设计信息、探测信息等
    使用配置化查询，确保健壮性和本体无关性

    Args:
        scheme_id: 紧急响应措施的业务ID（node_id属性，如"node_004391"）
        graph: Neo4j图对象，如果为None则使用默认配置

    Returns:
        包含方案信息、关联风险、设计信息、探测信息的字典
    """
    if graph is None:
        graph = config.get_graph()

    try:
        from retrieval.core.association_config import AssociationQueryExecutor
        executor = AssociationQueryExecutor(graph)
        results = executor.execute_plan_queries(scheme_id)

        # 将配置化查询结果转换为原有格式（保持向后兼容）
        return _format_plan_results(results)

    except Exception as e:
        logger.error(f"查询紧急响应措施 node_id={scheme_id} 失败: {e}")
        return {}


def _format_plan_results(query_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    将配置化查询结果转换为原有格式
    方案关联检索结果 - 历史信息，添加 history_ 前缀

    Args:
        query_results: AssociationQueryExecutor返回的结果

    Returns:
        格式化后的结果字典
    """
    formatted = {}

    try:
        # 方案属性（应急响应措施字段，无前缀）
        if "plan_properties" in query_results:
            formatted["scheme"] = query_results["plan_properties"]

        # 风险类型（应急响应措施字段，无前缀）
        if "risk_types" in query_results and query_results["risk_types"]:
            formatted["risk_type"] = query_results["risk_types"][0].get("风险类型")

        # 历史处置案例（加 history_ 前缀）
        if "historical_cases" in query_results and query_results["historical_cases"]:
            formatted["history_case"] = query_results["historical_cases"][0]

        # 预警等级（历史信息，无 history_ 前缀，因为本身就表示历史）
        if "warning_grade" in query_results and query_results["warning_grade"]:
            formatted["warning_grade"] = query_results["warning_grade"]

        # 施工信息（加 history_ 前缀）
        if "construction_info" in query_results and query_results["construction_info"]:
            formatted["history_construction_info"] = query_results["construction_info"]

        # 探测方法（加 history_ 前缀）
        if "detection_methods" in query_results and query_results["detection_methods"]:
            formatted["history_detection_methods"] = query_results["detection_methods"]

        # 探测结论（加 history_ 前缀）
        if "detection_conclusions" in query_results and query_results["detection_conclusions"]:
            formatted["history_detection_conclusions"] = query_results["detection_conclusions"]

        # 地质风险等级（加 history_ 前缀）
        if "geological_risk_levels" in query_results and query_results["geological_risk_levels"]:
            formatted["history_geological_risk_levels"] = query_results["geological_risk_levels"]

        # 围岩等级（加 history_ 前缀）
        if "rock_grades" in query_results and query_results["rock_grades"]:
            formatted["history_rock_grades"] = query_results["rock_grades"]

        # 时间信息（加 history_ 前缀）
        if "time_info" in query_results and query_results["time_info"]:
            formatted["history_time_info"] = query_results["time_info"]

        # 设计信息（加 history_ 前缀）
        if "design_info" in query_results and query_results["design_info"]:
            formatted["history_design_info"] = query_results["design_info"]

        return formatted

    except Exception as e:
        logger.error(f"格式化结果失败: {e}")
        return query_results  # 返回原始结果


def kg_mileage_relevance_retrieval(
    line_name: Optional[str],
    mileage: Optional[str],
    risk_type: Optional[str],
    tbm_name: str = "XX隧道",
    graph: Optional[object] = None
) -> Dict[str, Any]:
    """
    根据里程、线路、风险类型查询相关的设计信息和探测信息
    使用配置化查询，确保健壮性和本体无关性

    Args:
        line_name: 线路名称（左线/右线）
        mileage: 里程字符串（如"DK12+100"）
        risk_type: 风险类型
        tbm_name: 隧道名称
        graph: Neo4j图对象

    Returns:
        包含设计信息和探测信息的字典
    """
    if graph is None:
        graph = config.get_graph()

    if not mileage or not line_name or not risk_type:
        logger.info("线路、里程、风险类型存在空值")
        return {}

    # 解析里程
    try:
        mileage_start, mileage_end, mileage_prefix = parse_mileage(mileage)
    except Exception as e:
        logger.error(f"里程转换失败: {e}")
        return {}

    try:
        from retrieval.core.association_config import AssociationQueryExecutor
        executor = AssociationQueryExecutor(graph)
        results = executor.execute_mileage_queries(line_name, mileage_start, mileage_end, mileage_prefix, risk_type)

        # 将配置化查询结果转换为原有格式（保持向后兼容）
        return _format_mileage_results(results)

    except Exception as e:
        logger.error(f"里程关联检索失败: {e}")
        return {}


def _format_mileage_results(query_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    将配置化里程查询结果转换为原有格式
    里程关联检索结果 - 当前信息，添加 current_ 前缀

    Args:
        query_results: AssociationQueryExecutor返回的结果

    Returns:
        格式化后的结果字典
    """
    formatted = {}

    try:
        # 施工信息（加 current_ 前缀）
        if "construction_info" in query_results and query_results["construction_info"]:
            formatted["current_construction_info"] = query_results["construction_info"]

        # 设计信息（加 current_ 前缀）
        if "design_info" in query_results and query_results["design_info"]:
            formatted["current_design_info"] = query_results["design_info"]

        # 探测方法（加 current_ 前缀）
        if "detection_methods" in query_results and query_results["detection_methods"]:
            formatted["current_detection_methods"] = query_results["detection_methods"]

        # 探测结论（加 current_ 前缀）
        if "detection_conclusions" in query_results and query_results["detection_conclusions"]:
            formatted["current_detection_conclusions"] = query_results["detection_conclusions"]

        # 地质风险等级（加 current_ 前缀）
        if "geological_risk_levels" in query_results and query_results["geological_risk_levels"]:
            formatted["current_geological_risk_levels"] = query_results["geological_risk_levels"]

        # 围岩等级（加 current_ 前缀）
        if "rock_grades" in query_results and query_results["rock_grades"]:
            formatted["current_rock_grades"] = query_results["rock_grades"]

        # 历史处置案例（加 current_ 前缀）
        if "historical_cases" in query_results and query_results["historical_cases"]:
            formatted["current_historical_cases"] = query_results["historical_cases"]

        # 预警等级（加 current_ 前缀）
        if "warning_grades" in query_results and query_results["warning_grades"]:
            formatted["current_warning_grades"] = query_results["warning_grades"]

        # 风险评估（加 current_ 前缀）
        if "risk_assessments" in query_results and query_results["risk_assessments"]:
            formatted["current_risk_assessments"] = query_results["risk_assessments"]

        return formatted

    except Exception as e:
        logger.error(f"格式化里程结果失败: {e}")
        return query_results  # 返回原始结果
