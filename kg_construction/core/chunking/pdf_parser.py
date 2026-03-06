import os
from pathlib import Path
from typing import List, Dict, Optional


def is_pdf_file(filepath: str) -> bool:
    file_ext = Path(filepath).suffix.lower()
    return file_ext == '.pdf'


class PDFParser:
    def __init__(self):
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
        except ImportError:
            raise ImportError("请安装 pdfplumber: pip install pdfplumber")

    def extract_text(self, pdf_path: str) -> str:
        if not is_pdf_file(pdf_path):
            raise ValueError(f"文件不是PDF格式: {pdf_path}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        text_parts = []
        with self.pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n\n".join(text_parts)

    def extract_text_by_page(self, pdf_path: str) -> List[Dict[str, any]]:
        if not is_pdf_file(pdf_path):
            raise ValueError(f"文件不是PDF格式: {pdf_path}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        pages_data = []
        with self.pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    pages_data.append({
                        "page_number": page_num,
                        "text": page_text
                    })

        return pages_data

    def extract_text_with_structure(self, pdf_path: str) -> Dict[str, any]:
        if not is_pdf_file(pdf_path):
            raise ValueError(f"文件不是PDF格式: {pdf_path}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        full_text = []
        pages_data = []

        with self.pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
                    pages_data.append({
                        "page_number": page_num,
                        "text": page_text
                    })

        return {
            "full_text": "\n\n".join(full_text),
            "pages": pages_data,
            "total_pages": total_pages
        }
