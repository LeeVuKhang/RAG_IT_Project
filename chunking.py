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
            chunk_size=450,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        for i, item in enumerate(data):
            content = item.get("content", "").strip()
            header = item.get("header", "").strip()

            if not content or len(content.split()) < 30:
                continue

            split_chunks = splitter.split_text(content)

            for j, chunk in enumerate(split_chunks):
                if len(chunk.split()) < 20:
                    continue
                final_chunk = f"{header}\n{chunk}" if header else chunk

                chunk_meta = {
                    "id": f"chunk_{i}_{j}",
                    "title": item.get("title"),
                    "header": header,
                    "chunk": final_chunk,
                    "hash": Chunker._sha1(final_chunk)
                }
                chunks.append(chunk_meta)

        return chunks
