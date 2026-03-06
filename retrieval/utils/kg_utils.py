import json
import re
import logging
from typing import Optional, Tuple, Dict, Any, List

from retrieval.utils import config

logger = logging.getLogger(__name__)


def parse_mileage(mileage_str: str) -> Tuple[float, float]:
    """
    解析里程字符串，支持格式：
    - 起止格式 "起:DK13+298止:DK13+335"
    - 区间格式 "DyK1014+531.00~DyK1014+555.00" 或 "DyK1014+531.00～DyK1014+555.00"
    - 单点格式 "DK13+710"
    - 复合格式 "DK13+200~DK13+300"（设计信息的chainage字段）

    返回 (start, end)，单点时 start == end
    抛出 ValueError 当格式不匹配
    """

    def extract_single_mileage(km_str: str) -> float:
        # 兼容 DK、DyK、K 等开头，提取数字部分
        match = re.search(r'[a-zA-Z]*?(\d+)\+([\d.]+)', km_str)
        if not match:
            raise ValueError(f"无法解析的里程格式: {km_str}")
        km = int(match.group(1))
        m = float(match.group(2))
        return km * 1000 + m

    mileage_str = mileage_str.strip()

    # 类型1: 起止格式 "起:DK13+298止:DK13+335"
    m = re.match(r'起:([a-zA-Z0-9\+\.\s]+)止:([a-zA-Z0-9\+\.\s]+)', mileage_str)
    if m:
        start = extract_single_mileage(m.group(1).strip())
        end = extract_single_mileage(m.group(2).strip())
        return start, end

    # 类型2: 区间范围 "DyK1014+531.00~DyK1014+555.00" 或 "DyK1014+531.00～DyK1014+555.00"
    if "~" in mileage_str or "～" in mileage_str:
        # 统一分隔符
        normalized = mileage_str.replace("～", "~")
        parts = normalized.split("~")
        if len(parts) == 2:
            start = extract_single_mileage(parts[0].strip())
            end = extract_single_mileage(parts[1].strip())
            return start, end
        else:
            raise ValueError(f"无法解析的区间格式: {mileage_str}")

    # 类型3: 单点 "DK13+710"
    val = extract_single_mileage(mileage_str)
    return val, val


def extract_key_spa(text: str) -> Tuple[Optional[str], str]:
    """
    从文本中提取空间位置信息

    Args:
        text: 输入文本

    Returns:
        (mileage, line_name): 里程和线路名称
    """
    # 1. 匹配里程信息，例如 DK12+106.00、K23+800、23+120 等
    mileage_pattern = r'(?:D?K)?\d+\+\d+(?:\.\d+)?'
    mileage_match = re.search(mileage_pattern, text)
    key_spa = mileage_match.group() if mileage_match else None

    # 2. 匹配线路名称（左线/右线）
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
    scheme_id: str,
    graph: Optional[object] = None
) -> Dict[str, Any]:
    """
    根据紧急响应措施ID查询关联的历史风险、设计信息、探测信息等
    严格适配新版本KG结构

    Args:
        scheme_id: 紧急响应措施的node_id
        graph: Neo4j图对象，如果为None则使用默认配置

    Returns:
        包含方案信息、关联风险、设计信息、探测信息的字典
    """
    if graph is None:
        graph = config.get_graph()

    query = """
    MATCH (solution:紧急响应措施 {node_id: $scheme_id})
    MATCH (solution)-[:RESPONDS_TO]->(history:历史处置案例)

    // 获取预警等级
    OPTIONAL MATCH (history)-[:HAS_WARNING_GRADE]->(warning_grade:预警等级)

    // 获取风险类型
    OPTIONAL MATCH (history)-[:HAS_RISK_TYPE]->(risk_type:风险类型)

    // 获取发生时间和位置
    OPTIONAL MATCH (history)-[:OCCURS_AT]->(construction:施工信息)
    OPTIONAL MATCH (construction)-[:HAS_SPATIOTEMPORAL]->(time_node:时间)

    // 获取探测信息（通过施工信息→探测方法→探测结论）
    OPTIONAL MATCH (construction)-[:WAS_SURVEYED_BY]->(detection_method:探测方法)
    OPTIONAL MATCH (detection_method)-[:INDICATES]->(detection_conclusion:探测结论)
    OPTIONAL MATCH (detection_method)-[:INDICATES]->(geo_risk_level:地质风险等级)
    OPTIONAL MATCH (detection_method)-[:INDICATES]->(rock_grade:围岩等级)

    // 获取设计信息（通过施工信息→设计信息）
    OPTIONAL MATCH (construction)-[:IS_ASSOCIATED_WITH]->(design:设计信息)
    OPTIONAL MATCH (design)-[:HAS_SURROUNDING_ROCK_GRADE]->(design_rock_grade:围岩等级)

    RETURN
        solution.node_id AS solution_id,
        solution.riskType AS solution_risk_type,
        solution.applicableConditions AS applicable_conditions,
        solution.emergencyResponseGuidelines AS emergency_response_guidelines,
        history.node_id AS history_id,
        history.s_id AS history_s_id,
        history.riskDescription AS risk_description,
        history.warningDate AS warning_date,
        history.chainage AS history_chainage,
        warning_grade.warningGrade AS warning_grade,
        risk_type.riskType AS risk_type_node,
        construction.node_id AS construction_id,
        construction.chainage AS construction_chainage,
        construction.information AS construction_info,
        time_node.time AS warning_time,
        detection_method.node_id AS detection_method_id,
        detection_method.detectionMethod AS detection_method,
        detection_method.chainage AS detection_chainage,
        detection_conclusion.detectionConclusion AS detection_conclusion,
        detection_conclusion.geologicalElements AS geological_elements,
        detection_conclusion.后续建议 AS detection_suggestions,
        geo_risk_level.geologicalRiskGrade AS geological_risk_level,
        rock_grade.grade AS rock_grade_from_detection,
        design.node_id AS design_id,
        design.chainage AS design_chainage,
        design.information AS design_info,
        design.length AS design_length,
        design.grade AS design_grade_inline,
        design_rock_grade.grade AS design_rock_grade
    """

    try:
        results = graph.run(query, scheme_id=scheme_id).data()
    except Exception as e:
        logger.error(f"查询紧急响应措施 {scheme_id} 失败: {e}")
        return {}

    if not results:
        logger.warning(f"未找到紧急响应措施 {scheme_id} 的关联信息")
        return {}

    record = results[0]

    try:
        # 构建方案信息
        plan_data = {
            "riskType": record.get("solution_risk_type"),
            "applicableConditions": record.get("applicable_conditions"),
            "emergencyResponseGuidelines": record.get("emergency_response_guidelines")
        }

        # 构建关联风险信息
        risk_data = {
            "riskDescription": record.get("risk_description"),
            "warningDate": record.get("warning_date"),
            "chainage": record.get("history_chainage") or record.get("construction_chainage"),
            "预警等级": record.get("warning_grade"),
            "风险类型": record.get("risk_type_node"),
            "发生时间": record.get("warning_time"),
            "发生位置": {
                "chainage": record.get("construction_chainage") or record.get("history_chainage"),
                "information": record.get("construction_info"),
                "探测信息": {
                    "detectionMethod": record.get("detection_method"),
                    "chainage": record.get("detection_chainage"),
                    "detectionConclusion": record.get("detection_conclusion"),
                    "geologicalElements": record.get("geological_elements"),
                    "后续建议": record.get("detection_suggestions"),
                    "地质风险等级": record.get("geological_risk_level"),
                    "围岩等级": record.get("rock_grade_from_detection")
                },
                "设计信息": {
                    "chainage": record.get("design_chainage"),
                    "length": record.get("design_length"),
                    "information": record.get("design_info"),
                    "grade": record.get("design_rock_grade") or record.get("design_grade_inline"),
                    "note": "围岩等级优先使用关联关系中的，其次使用设计信息内联属性"
                }
            }
        }

        return {
            "方案": plan_data,
            "关联风险": risk_data
        }

    except Exception as e:
        logger.error(f"解析查询结果失败: {e}")
        return {}


def kg_mileage_relevance_retrieval(
    line_name: Optional[str],
    mileage: Optional[str],
    risk_type: Optional[str],
    tbm_name: str = "XX隧道",
    graph: Optional[object] = None
) -> Dict[str, Any]:
    """
    根据里程、线路、风险类型查询相关的设计信息和探测信息
    严格适配新版本KG结构

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
        trans_mileage, _ = parse_mileage(mileage)
    except Exception as e:
        logger.error(f"里程转换失败: {e}")
        return {}

    output = {}

    # 1. 查询设计信息
    # 新版本：设计信息有chainage字段（格式："起始里程～终止里程"）
    design_query = """
    MATCH (design:设计信息)
    WHERE design.chainage IS NOT NULL
    RETURN design
    """

    try:
        all_design_results = graph.run(design_query).data()
    except Exception as e:
        logger.error(f"设计节点查询失败: {e}")
        return output

    # 查找匹配的设计信息
    for record in all_design_results:
        design_data = record.get("design", {})

        if not design_data:
            continue

        try:
            # 解析设计信息的chainage（"起始里程～终止里程"）
            design_chainage = design_data.get("chainage", "")
            if design_chainage:
                start, end = parse_mileage(design_chainage)
                if start <= trans_mileage < end:
                    # 找到匹配的设计信息
                    output["设计信息"] = {
                        "chainage": design_chainage,
                        "length": design_data.get("length"),
                        "information": design_data.get("information"),
                        "grade": design_data.get("grade"),
                        "note": "设计信息的内联围岩等级属性"
                    }
                    break
        except Exception as e:
            logger.error(f"设计信息解析失败: {e}")
            continue

    # 2. 查询探测信息
    # 新版本：探测方法有chainage字段，通过INDICATES关系指向探测结论
    detection_query = """
    MATCH (detection_method:探测方法)-[:INDICATES]->(detection_conclusion:探测结论)
    WHERE detection_method.chainage IS NOT NULL
    RETURN detection_method, detection_conclusion
    """

    try:
        all_detection_results = graph.run(detection_query).data()
    except Exception as e:
        logger.error(f"探测节点查询失败: {e}")
        return output

    for record in all_detection_results:
        detection_method_data = record.get("detection_method", {})
        detection_conclusion_data = record.get("detection_conclusion", {})

        try:
            detection_chainage = detection_method_data.get("chainage", "")
            if detection_chainage:
                start, end = parse_mileage(detection_chainage)
                if start <= trans_mileage < end:
                    # 获取该探测方法指示的所有信息（地质风险等级、围岩等级等）
                    detection_id = detection_method_data.get("node_id")

                    related_query = """
                    MATCH (detection_method:探测方法 {node_id: $detection_id})-[:INDICATES]->(target)
                    RETURN target
                    """
                    related_results = graph.run(related_query, detection_id=detection_id).data()

                    # 提取所有关联信息
                    geo_risk_level = None
                    rock_grade = None
                    for related_record in related_results:
                        target = related_record.get("target", {})
                            # 地质风险等级
                        if target.get("geologicalRiskGrade"):
                            geo_risk_level = target.get("geologicalRiskGrade")
                        # 围岩等级
                        elif target.get("grade"):
                            existing_grade = rock_grade
                            new_grade = target.get("grade")
                            # 如果已有围岩等级，优先保留
                            if not existing_grade:
                                rock_grade = new_grade

                    output["探测信息"] = {
                        "detectionMethod": detection_method_data.get("detectionMethod"),
                        "chainage": detection_chainage,
                        "detectionConclusion": detection_conclusion_data.get("detectionConclusion"),
                        "geologicalElements": detection_conclusion_data.get("geologicalElements"),
                        "后续建议": detection_conclusion_data.get("后续建议"),
                        "地质风险等级": geo_risk_level,
                        "围岩等级": rock_grade
                    }
                    break
        except Exception as e:
            logger.error(f"探测信息解析失败: {e}")
            continue

    return output
