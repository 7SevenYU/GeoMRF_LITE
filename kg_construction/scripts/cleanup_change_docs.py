"""
清理变更纪要的提取结果，准备重新提取
"""
import os
import glob
from pathlib import Path

def cleanup_change_documents():
    """清理变更纪要的提取结果文件"""

    # 项目根目录
    project_root = Path(__file__).parent.parent.parent

    # 变更纪要提取结果目录
    extraction_results_dir = project_root / "kg_construction" / "data" / "processed" / "extraction_results" / "变更纪要"

    if extraction_results_dir.exists():
        # 删除所有JSON文件
        json_files = list(extraction_results_dir.glob("*.json"))
        for json_file in json_files:
            try:
                os.remove(json_file)
                print(f"已删除: {json_file.name}")
            except Exception as e:
                print(f"删除失败 {json_file.name}: {e}")

        print(f"\n清理完成！共删除 {len(json_files)} 个文件")
        print("现在可以重新运行 extract_triples.py 来提取变更纪要")
    else:
        print(f"目录不存在: {extraction_results_dir}")

if __name__ == "__main__":
    cleanup_change_documents()