import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any

# 智能检测项目根目录，支持两种运行方式
script_dir = Path(__file__).parent
if script_dir.name == "scripts":
    # 从项目根目录运行：python -m kg_construction.scripts.chunk_documents
    project_root = script_dir.parent.parent
else:
    # 在scripts目录直接运行：python chunk_documents.py（调试模式）
    project_root = script_dir.parent.parent

# 确保项目根目录在sys.path中
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from kg_construction.core.chunking.text_chunker import TextChunker
from kg_construction.core.chunking.pdf_parser import PDFParser
from kg_construction.core.chunking.document_classifier import DocumentClassifier
from kg_construction.core.chunking.chunk_id_generator import ChunkIDGenerator
from kg_construction.utils.prompt_loader import PromptLoader
from kg_construction.utils.logger import setup_logger


class DocumentChunker:
    def __init__(self, source_dir: str, output_dir: str):
        # 将相对路径转换为基于project_root的绝对路径
        if not Path(source_dir).is_absolute():
            source_dir = project_root / source_dir
        if not Path(output_dir).is_absolute():
            output_dir = project_root / output_dir

        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.logger = setup_logger("DocumentChunker", "chunking.log")

        self.text_chunker = TextChunker(chunk_size=4096, overlap=200)
        self.pdf_parser = PDFParser()
        self.doc_classifier = DocumentClassifier()
        self.prompt_loader = PromptLoader()
        self.chunk_id_gen = ChunkIDGenerator(str(self.output_dir))

        (self.output_dir / "chunks").mkdir(parents=True, exist_ok=True)

    def process_all_documents(self):
        self.logger.info(f"开始处理文档，源目录: {self.source_dir}")

        all_results = {}

        for doc_type in ["变更纪要", "超前地质预报", "处置记录", "设计信息", "专家经验"]:
            result = self.process_document_type(doc_type)
            if result:
                all_results[doc_type] = result

        self._save_summary(all_results)
        self.logger.info("文档分块完成")

        return all_results

    def process_document_type(self, doc_type: str) -> Dict[str, Any]:
        self.logger.info(f"处理文档类型: {doc_type}")

        source_type_dir = self.source_dir / doc_type
        if not source_type_dir.exists():
            self.logger.warning(f"目录不存在: {source_type_dir}")
            return None

        all_chunks = []
        processed_count = 0
        skipped_count = 0

        # 特殊处理：超前地质预报需要递归查找子目录
        if doc_type == "超前地质预报":
            json_files = list(source_type_dir.glob("**/*.json"))
            pdf_files = list(source_type_dir.glob("**/*.pdf"))
        else:
            pdf_files = list(source_type_dir.glob("*.pdf"))
            json_files = list(source_type_dir.glob("*.json"))

        self.logger.info(f"  找到 {len(pdf_files)} 个PDF文件, {len(json_files)} 个JSON文件")

        # 对PDF文件使用DocumentClassifier识别实际类型
        for pdf_file in pdf_files:
            file_path = str(pdf_file)

            # 检查是否已处理
            if self.chunk_id_gen.is_processed(file_path):
                chunk_count = self.chunk_id_gen.get_file_chunk_count(file_path)
                self.logger.info(f"  跳过已处理文件: {pdf_file.name} ({chunk_count} chunks)")
                skipped_count += 1
                continue

            actual_doc_type = self.doc_classifier.classify(file_path)
            chunks = self._process_pdf(pdf_file, actual_doc_type)

            if chunks:
                all_chunks.extend(chunks)
                # 标记文件为已处理
                self.chunk_id_gen.mark_file_processed(file_path, len(chunks))
                processed_count += 1

        # 对JSON文件使用DocumentClassifier识别实际类型
        for json_file in json_files:
            file_path = str(json_file)

            # 检查是否已处理
            if self.chunk_id_gen.is_processed(file_path):
                chunk_count = self.chunk_id_gen.get_file_chunk_count(file_path)
                self.logger.info(f"  跳过已处理文件: {json_file.name} ({chunk_count} chunks)")
                skipped_count += 1
                continue

            actual_doc_type = self.doc_classifier.classify(file_path)
            chunks = self._process_json(json_file, actual_doc_type)

            if chunks:
                all_chunks.extend(chunks)
                # 标记文件为已处理
                self.chunk_id_gen.mark_file_processed(file_path, len(chunks))
                processed_count += 1

        self.logger.info(f"  处理完成: 新处理 {processed_count} 个文件, 跳过 {skipped_count} 个已处理文件")

        if not all_chunks:
            return None

        # 按实际文档类型分组保存
        chunks_by_type_and_format = {}
        for chunk in all_chunks:
            actual_type = chunk.get("document_type", doc_type)
            file_type = chunk.get("file_type", "unknown")  # json or pdf

            # 创建组合键：类型_格式
            key = f"{actual_type}({file_type})" if file_type != "unknown" else actual_type

            if key not in chunks_by_type_and_format:
                chunks_by_type_and_format[key] = {
                    "chunks": [],
                    "doc_type": actual_type,
                    "file_type": file_type
                }
            chunks_by_type_and_format[key]["chunks"].append(chunk)

        # 为每种类型和格式保存单独的文件
        results = {}
        for key, data in chunks_by_type_and_format.items():
            type_chunks = data["chunks"]
            actual_type = data["doc_type"]
            file_type = data["file_type"]

            if type_chunks:
                # 根据文件格式调整输出文件名
                if file_type == "json":
                    output_filename = f"{actual_type}(JSON)_chunks.json"
                elif file_type == "pdf":
                    output_filename = f"{actual_type}(PDF)_chunks.json"
                else:
                    output_filename = f"{actual_type}_chunks.json"

                output_file = self.output_dir / "chunks" / output_filename
                self._save_chunks(type_chunks, output_file, actual_type)
                self.logger.info(f"  {actual_type}({file_type}): 生成了 {len(type_chunks)} 个文本块")
                results[key] = {
                    "document_type": actual_type,
                    "file_type": file_type,
                    "total_chunks": len(type_chunks),
                    "output_file": str(output_file)
                }

        return results if results else None

    def _process_pdf(self, pdf_file: Path, doc_type: str) -> List[Dict[str, Any]]:
        try:
            self.logger.info(f"    处理PDF: {pdf_file.name}")

            doc_config = self.prompt_loader.get_document_config(doc_type)
            self.logger.info(f"    doc_type: {doc_type}")
            self.logger.info(f"    doc_config keys: {list(doc_config.keys())}")

            segmentation = doc_config.get("segmentation", {})
            strategy = segmentation.get("strategy", "regular")
            self.logger.info(f"    segmentation strategy: {strategy}")

            if strategy == "title_based":
                return self._process_title_based(pdf_file, doc_type, segmentation)
            else:
                self.logger.info(f"    使用 regular 分块策略")
                return self._process_regular(pdf_file, doc_type)

        except Exception as e:
            self.logger.error(f"    处理PDF失败 {pdf_file}: {e}")
            return []

    def _process_regular(self, pdf_file: Path, doc_type: str) -> List[Dict[str, Any]]:
        pages_data = self.pdf_parser.extract_text_by_page(str(pdf_file))
        chunks = self.text_chunker.chunk_text_by_pages(
            pages_data,
            metadata={
                "document_type": doc_type,
                "source_file": str(pdf_file),
                "file_type": "pdf"
            }
        )

        for chunk in chunks:
            chunk["chunk_id"] = self.chunk_id_gen.get_next_chunk_id()

        return chunks

    def _process_title_based(self, pdf_file: Path, doc_type: str, segmentation: Dict) -> List[Dict[str, Any]]:
        """基于标题的分段处理（通用）"""
        try:
            self.logger.info(f"      使用title_based分段处理: {pdf_file.name}")

            pdf_text = self.pdf_parser.extract_text(str(pdf_file))

            namespace = segmentation.get("namespace")
            section_patterns = segmentation.get("section_patterns", {})

            segments = self._segment_by_patterns(pdf_text, section_patterns)

            if not segments:
                self.logger.warning(f"      分段失败，回退到普通分块方式")
                return self._process_regular(pdf_file, doc_type)

            chunks = []
            valid_sections = self.prompt_loader.get_document_sections(doc_type)

            for section_num in valid_sections:
                if section_num not in segments:
                    self.logger.warning(f"      段{section_num}缺失，跳过")
                    continue

                section_config = self.prompt_loader.get_section_config(doc_type, section_num)

                chunk = {
                    "chunk_id": self.chunk_id_gen.get_next_chunk_id(),
                    "text": segments[section_num],
                    "document_type": doc_type,
                    "source_file": str(pdf_file),
                    "file_type": "pdf",
                    "metadata": {
                        "segmentation": {
                            "strategy": "title_based",
                            "namespace": namespace,
                            "section_id": section_num,
                            "section_name": section_config["name"]
                        },
                        "skip_save": True
                    }
                }

                chunks.append(chunk)
                self.logger.info(f"      创建分段chunk: {chunk['chunk_id']} ({len(segments[section_num])} 字符)")

            self.logger.info(f"      分段完成，生成 {len(chunks)} 个分段chunk")
            return chunks

        except Exception as e:
            self.logger.error(f"      处理失败 {pdf_file}: {e}")
            return self._process_regular(pdf_file, doc_type)

    def _extract_detection_method_from_path(self, json_file: Path, doc_type: str) -> str:
        """从文件路径中提取探测方法类型（仅适用于超前地质预报）"""
        if not doc_type.startswith("超前地质预报"):
            return ""

        try:
            # 获取相对路径中的二级目录名
            # 例如：超前地质预报/水平声波剖面/工区/数据.json
            # 需要提取 "水平声波剖面"
            path_parts = json_file.parts
            for i, part in enumerate(path_parts):
                if part == "超前地质预报" and i + 1 < len(path_parts):
                    return path_parts[i + 1]
            return ""
        except Exception as e:
            self.logger.warning(f"Failed to extract detection method from path {json_file}: {e}")
            return ""

    def _segment_by_patterns(self, text: str, section_patterns: Dict[str, str]) -> Dict[str, str]:
        """按配置的模式将文本分段（通用）"""
        if not text:
            return {}

        text = text.replace("．", ".").replace("、", ".")
        text = re.sub(r'\s+', ' ', text)

        positions = {}
        for section_num, pattern in section_patterns.items():
            match = re.search(pattern, text)
            if match:
                positions[section_num] = match.start()
                self.logger.info(f"      找到章节标记: 段{section_num} (位置: {match.start()})")

        if not positions:
            self.logger.warning("      未找到任何章节标记")
            return {}

        segments = {}
        sorted_sections = sorted(positions.items(), key=lambda x: x[1])

        if sorted_sections:
            first_pos = sorted_sections[0][1]
            segments["0"] = text[:first_pos].strip()
            self.logger.info(f"      提取段0（开头）: {len(segments['0'])} 字符")

        for i, (section_num, pos) in enumerate(sorted_sections):
            if i + 1 < len(sorted_sections):
                next_pos = sorted_sections[i + 1][1]
                segments[section_num] = text[pos:next_pos].strip()
            else:
                segments[section_num] = text[pos:].strip()
            self.logger.info(f"      提取段{section_num}: {len(segments[section_num])} 字符")

        return segments


    def _process_json(self, json_file: Path, doc_type: str) -> List[Dict[str, Any]]:
        try:
            self.logger.info(f"    处理JSON: {json_file.name}")

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = [data]

            # 提取探测方法（仅针对超前地质预报）
            detection_method = self._extract_detection_method_from_path(json_file, doc_type)

            chunks = []
            for record in data:
                # 特殊处理：洞身纵向地质素描需要预处理数组格式
                if detection_method == "洞身纵向地质素描" and isinstance(record, dict):
                    # 将"结论"数组转换为对象格式
                    if "结论" in record and isinstance(record["结论"], list):
                        conclusion_obj = {}
                        for item in record["结论"]:
                            if isinstance(item, dict):
                                conclusion_obj.update(item)
                        record["结论"] = conclusion_obj

                    # 将"施工揭示"数组转换为对象格式（可选）
                    if "施工揭示" in record and isinstance(record["施工揭示"], list):
                        reveal_obj = {}
                        for item in record["施工揭示"]:
                            if isinstance(item, dict):
                                reveal_obj.update(item)
                        record["施工揭示"] = reveal_obj

                chunk = {
                    "chunk_id": self.chunk_id_gen.get_next_chunk_id(),
                    "text": json.dumps(record, ensure_ascii=False),
                    "document_type": doc_type,
                    "source_file": str(json_file),
                    "file_type": "json",
                    "metadata": {
                        "record_data": record
                    }
                }

                if detection_method:
                    chunk["metadata"]["detection_method"] = detection_method

                chunks.append(chunk)

            return chunks

        except Exception as e:
            self.logger.error(f"    处理JSON失败 {json_file}: {e}")
            return []

    def _save_chunks(self, chunks: List[Dict], output_file: Path, doc_type: str):
        output_data = {
            "document_type": doc_type,
            "total_chunks": len(chunks),
            "chunks": chunks
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"    保存分块结果: {output_file}")

    def _save_summary(self, all_results: Dict[str, Any]):
        summary_file = self.output_dir / "chunks" / "chunking_summary.json"

        summary = {
            "total_document_types": len(all_results),
            "total_chunks": sum(r.get("total_chunks", 0) for r in all_results.values() if r),
            "details": all_results
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.logger.info(f"保存分块摘要: {summary_file}")


if __name__ == "__main__":
    # 使用相对路径，基于项目根目录
    source_directory = "kg_construction/data/source"
    output_directory = "kg_construction/data/processed"

    chunker = DocumentChunker(source_directory, output_directory)
    chunker.process_all_documents()

    print("\n文档分块完成！")
    print(f"结果保存在: {output_directory}/chunks/")
