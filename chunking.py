import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chunker:
    @staticmethod
    def _sha1(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def make_chunks(data: list) -> list:
        chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        for i, item in enumerate(data):
            content = item.get("content", "").strip()
            if not content or len(content) < 20:
                continue

            split_chunks = splitter.split_text(content)

            for j, chunk in enumerate(split_chunks):
                chunk_meta = {
                    "id": f"chunk_{i}_{j}",
                    "title": item.get("title"),
                    "header": item.get("header"),
                    "chunk": chunk,
                    "hash": Chunker._sha1(chunk)
                }
                chunks.append(chunk_meta)

        return chunks
