import json
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Setup device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("💻 Đang chạy trên:", device)
if device == "cuda":
    print("🚀 GPU:", torch.cuda.get_device_name(0))
torch.cuda.empty_cache()

# --- Load model ---
model_name = "AITeamVN/Vietnamese_Embedding"
model = SentenceTransformer(model_name, device=device)

# --- Load chunks ---
file_path = r"D:\Đại học\Năm 3\IT Project\RAG_IT_PROJECT\dulichvn_chunks.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Lọc dữ liệu: bỏ chunk ngắn ---
min_words = 20
filtered_data = [item for item in data if "chunk" in item and len(item["chunk"].split()) >= min_words]

texts = [item["chunk"] for item in filtered_data]
print(f"📄 Tổng số chunks: {len(data)}")
print(f"✅ Sau khi lọc còn: {len(texts)} (>= {min_words} từ)")

# --- Tạo embeddings ---
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# --- Tạo FAISS index ---
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype="float32"))

# --- Lưu FAISS + Metadata ---
faiss_file = "embeddings.faiss"
faiss.write_index(index, faiss_file)
print(f"💾 Đã lưu FAISS index vào: {faiss_file}")
print(f"🔹 Số vector: {index.ntotal}, Số chiều: {dim}")

metadata_file = "metadata.json"
with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
print(f"💾 Đã lưu metadata vào: {metadata_file}")
