import json
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Embedder:
    @staticmethod
    def setup_device():
        """Thiết lập thiết bị (CPU/GPU)"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Đang chạy trên:", device)
        if device == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        return device

    @staticmethod
    def load_model(device, model_name="AITeamVN/Vietnamese_Embedding"):
        """Load SentenceTransformer model"""
        return SentenceTransformer(model_name, device=device)

    @staticmethod
    def load_chunks(file_path):
        """Đọc file JSON chứa chunks"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @staticmethod
    def prepare_data(data):
        """Chuẩn bị dữ liệu (format embed_text, không lọc theo số từ)"""
        filtered_data = []
        for idx, item in enumerate(data):
            text = item.get("text") or item.get("chunk", "")
            embed_text = f"{item.get('title','')} - {item.get('header','')}\n{text}"
            item["chunk"] = text
            item["embed_text"] = embed_text
            item["vector_id"] = idx
            filtered_data.append(item)

        texts = [item["embed_text"] for item in filtered_data]
        print(f"Tổng số chunks: {len(data)}")
        print(f"Giữ lại toàn bộ: {len(texts)} chunks")
        return filtered_data, texts

    @staticmethod
    def build_embeddings(model, texts, batch_size=32):
        """Sinh embeddings từ model"""
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings

    @staticmethod
    def build_faiss_index(embeddings, faiss_file="embeddings.faiss"):
        """Tạo FAISS index và lưu ra file"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype="float32"))
        faiss.write_index(index, faiss_file)
        print(f"💾 Đã lưu FAISS index vào: {faiss_file}")
        print(f"🔹 Số vector: {index.ntotal}, Số chiều: {dim}")
        return index

    @staticmethod
    def save_metadata(filtered_data, metadata_file="metadata.json"):
        """Lưu metadata mapping vector_id -> thông tin gốc"""
        metadata = []
        for item in filtered_data:
            metadata.append({
                "vector_id": item["vector_id"],
                "id": item.get("id"),
                "title": item.get("title"),
                "header": item.get("header"),
                "chunk": item.get("chunk")
            })

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        print(f"Đã lưu metadata vào: {metadata_file}")
