import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2

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

dim = embeddings.shape[1]
print(f"🔹 Số vector: {len(embeddings)}, Số chiều: {dim}")

# --- Kết nối PostgreSQL ---
conn = psycopg2.connect(
    dbname="vectordb",
    user="postgres",
    password="",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# --- Lưu vào DB ---
for text, meta, vector in zip(texts, filtered_data, embeddings):
    # Insert document
    cur.execute(
        """
        INSERT INTO documents (title, source, chunk)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (meta.get("title"), meta.get("source"), text)
    )
    doc_id = cur.fetchone()[0]

    # Insert embedding
    cur.execute(
        """
        INSERT INTO embeddings (document_id, embedding)
        VALUES (%s, %s)
        """,
        (doc_id, vector.tolist())
    )

conn.commit()
cur.close()
conn.close()

print("💾 Đã lưu toàn bộ embeddings + metadata vào PostgreSQL thành công!")
