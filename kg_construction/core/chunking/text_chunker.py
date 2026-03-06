import re
from datetime import datetime
from typing import List, Dict, Optional


class TextChunker:
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_sentences(self, text: str) -> List[str]:
        sentence_endings = re.compile(r'([。！？.!?]+)')
        sentences = sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk = ""
        current_length = 0
        char_start = 0
        page_start = 1

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "char_start": char_start,
                    "char_end": char_start + current_length,
                    "page_range": [page_start, page_start]
                })

                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                char_start += current_length - self.overlap if current_length > self.overlap else char_start
                current_length = len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length + 1

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "char_start": char_start,
                "char_end": char_start + current_length,
                "page_range": [page_start, page_start]
            })

        for idx, chunk in enumerate(chunks):
            chunk.update({
                "chunk_index": idx,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            if metadata:
                chunk.update(metadata)

        return chunks

    def chunk_text_by_pages(self, pages_data: List[Dict], metadata: Optional[Dict] = None) -> List[Dict]:
        all_chunks = []
        global_chunk_index = 0
        global_char_start = 0

        for page_data in pages_data:
            page_num = page_data.get("page_number", 1)
            page_text = page_data.get("text", "")

            page_chunks = self.chunk_text(page_text, metadata=None)

            for chunk in page_chunks:
                chunk["page_range"] = [page_num, page_num]
                chunk["chunk_index"] = global_chunk_index
                chunk["char_start"] = global_char_start
                chunk["char_end"] = global_char_start + len(chunk["text"])
                chunk["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if metadata:
                    chunk.update(metadata)

                all_chunks.append(chunk)
                global_chunk_index += 1
                global_char_start = chunk["char_end"]

        return all_chunks
