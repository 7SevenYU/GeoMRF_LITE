import os
from pathlib import Path
from typing import Optional


class DocumentClassifier:
    DOCUMENT_TYPE_MAPPING = {
        "变更纪要": "变更纪要",
        "变更会议纪要": "变更纪要",
        "会议纪要": "变更纪要",
        "超前地质预报": "超前地质预报",
        "地质预报": "超前地质预报",
        "预报": "超前地质预报",
        "处置记录": "处置记录",
        "风险处置": "处置记录",
        "处置": "处置记录",
        "专家经验": "专家经验",
        "专家": "专家经验",
        "设计信息": "设计信息",
        "设计": "设计信息",
        "设计说明": "设计信息"
    }

    def __init__(self):
        pass

    def classify_by_path(self, file_path: str) -> str:
        path_obj = Path(file_path)

        # 检查是否在"超前地质预报"目录下，支持二级子目录（探测方法类型）
        for parent in path_obj.parents:
            if parent.name == "超前地质预报":
                # 获取探测方法类型（二级目录名）
                parts = path_obj.relative_to(parent).parts
                if len(parts) >= 2:
                    detection_method = parts[0]  # 第一级子目录是探测方法
                    # 支持的探测方法类型
                    supported_methods = ["水平声波剖面", "TSP", "地质雷达", "超前钻探", "红外探水", "洞身纵向地质素描"]
                    if detection_method in supported_methods:
                        return f"超前地质预报_{detection_method}"
                    else:
                        # 默认使用水平声波剖面
                        return "超前地质预报_水平声波剖面"
                return "超前地质预报_水平声波剖面"

        # 其他文档类型的常规识别
        for parent in path_obj.parents:
            folder_name = parent.name
            if folder_name in self.DOCUMENT_TYPE_MAPPING:
                return self.DOCUMENT_TYPE_MAPPING[folder_name]

        file_name = path_obj.stem
        for keyword, doc_type in self.DOCUMENT_TYPE_MAPPING.items():
            if keyword in file_name:
                return doc_type

        return "未知类型"

    def classify_by_content(self, text: str, file_path: Optional[str] = None) -> str:
        if file_path:
            return self.classify_by_path(file_path)

        text_lower = text.lower()

        if any(keyword in text for keyword in ["变更", "会议纪要"]):
            return "变更纪要"
        elif any(keyword in text for keyword in ["地质预报", "超前预报", "探测"]):
            return "超前地质预报"
        elif any(keyword in text for keyword in ["处置", "风险处置", "应急"]):
            return "处置记录"

        return "未知类型"

    def classify(self, file_path: str, text: Optional[str] = None) -> str:
        return self.classify_by_path(file_path)
