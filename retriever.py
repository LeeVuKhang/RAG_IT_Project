import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import json


class Retriever:
    def __init__(self,
                 faiss_path="embeddings.faiss",
                 metadata_path="metadata.json",
                 embed_model="AITeamVN/Vietnamese_Embedding",
                 use_rerank=True,
                 rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Khởi tạo retriever với FAISS index, metadata, model embedding và optional reranker
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(embed_model, device=device)

        # Load FAISS index
        self.index = faiss.read_index(faiss_path)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.id2meta = {item["vector_id"]: item for item in metadata}

        # Reranker
        self.use_rerank = use_rerank
        if use_rerank:
            self.reranker = CrossEncoder(rerank_model, device=device)

    def embed_query(self, query: str):
        """
        Encode query thành vector embedding
        """
        query_vec = self.model.encode([query], normalize_embeddings=True)
        return np.array(query_vec, dtype="float32")

    def search(self, query: str, top_k: int = 10):
        """
        Tìm kiếm top_k kết quả gần nhất trong FAISS index + rerank nếu bật
        """
        query_vec = self.embed_query(query)
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.id2meta.get(idx)
            if meta:
                results.append({
                    "id": int(idx),
                    "score": float(dist),
                    "title": meta.get("title", ""),
                    "chunk": meta.get("chunk", "")
                })

        if self.use_rerank and results:
            pairs = [[query, r["chunk"]] for r in results]
            scores = self.reranker.predict(pairs)
            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
            results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        return results


# --- Test ---
if __name__ == "__main__":
    retriever = Retriever(
        faiss_path="embeddings.faiss",
        metadata_path="metadata.json",
        embed_model="AITeamVN/Vietnamese_Embedding",
        use_rerank=True
    )

    query = "Du lịch Nam Du"
    results = retriever.search(query, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['title']}]")
        print(f"   Score: {r['score']:.4f}")
        if "rerank_score" in r:
            print(f"   Rerank: {r['rerank_score']:.4f}")
        print(f"   Nội dung: {r['chunk']}\n")