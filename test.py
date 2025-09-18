import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ===== Load model và FAISS index =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("AITeamVN/Vietnamese_Embedding", device=device)

index = faiss.read_index("embeddings.faiss")

# Load metadata (giả sử bạn đã lưu ra file JSON khi tạo embeddings)
import json
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)  # List các dict: {"url":..., "title":..., "chunk":...}

# ===== Hàm search =====
def search(query, top_k=3):
    # Encode query
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    # Tìm kiếm trong FAISS
    distances, indices = index.search(query_vec, top_k)

    # Trả về kết quả
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata):
            results.append({
                "score": float(dist),
                "url": metadata[idx]["url"],
                "title": metadata[idx]["title"],
                "chunk": metadata[idx]["chunk"]
            })
    return results

# ===== Test query =====
query = "Địa điểm du lịch đẹp ở Hà Nội"
results = search(query, top_k=3)

# In kết quả
for i, r in enumerate(results, 1):
    print(f"{i}. [{r['title']}]({r['url']})")
    print(f"   Score: {r['score']:.4f}")
    print(f"   Nội dung: {r['chunk']}\n")