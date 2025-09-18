import json
import os
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"
print("💻 Đang chạy trên:", device)


model_name = "AITeamVN/Vietnamese_Embedding"
model = SentenceTransformer(model_name, device=device)
file_path = r"D:\Đại học\Năm 3\IT Project\Web Scraping\chunks.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["chunk"] for item in data if "chunk" in item]

print(f"📄 Số đoạn văn cần embedding: {len(texts)}")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

dim = embeddings.shape[1] 
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype="float32"))

faiss_file = "embeddings.faiss"
faiss.write_index(index, faiss_file)

print(f"Đã lưu FAISS index vào: {faiss_file}")
print(f"Số vector: {index.ntotal}, Số chiều: {dim}")