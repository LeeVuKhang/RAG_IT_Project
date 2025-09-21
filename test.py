import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("AITeamVN/Vietnamese_Embedding", device=device)

index = faiss.read_index("embeddings.faiss")

import json
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)  

def search(query, top_k=10):
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    distances, indices = index.search(query_vec, top_k)

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

query = "homestay sóc trăng"
results = search(query, top_k=10)

for i, r in enumerate(results, 1):
    print(f"{i}. [{r['title']}]({r['url']})")
    print(f"   Score: {r['score']:.4f}")
    print(f"   Nội dung: {r['chunk']}\n")