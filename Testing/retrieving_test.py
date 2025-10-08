# -*- coding: utf-8 -*-
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import json
import os
from typing import List, Dict, Any
import cohere

# ==============================================================================
# CONFIGURATION
# ==============================================================================
COHERE_API_KEY = "ws2oGGyOm5ewVRwiXqiLiugOzpkdckTwbu7Bo18w"

MODELS_TO_TEST = [
    "AITeamVN/Vietnamese_Embedding",
    "BAAI/bge-m3",
    "Cohere/embed-multilingual-v3.0",
    "jinaai/jina-embeddings-v3"
]

QUERIES_TO_TEST = [
    "Du lịch Nam Du",
    "Ẩm thực Sài Gòn"
]

TOP_K = 3
FAISS_DIR = "Testing/faiss"
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")
# ADDED: Tên file JSON đầu ra
OUTPUT_JSON_FILE = "retrieval_comparison_results.json"


class Retriever:
    """
    Class to load a FAISS index, metadata, and perform searches.
    """
    def __init__(self,
                 faiss_path: str,
                 metadata_path: str,
                 embed_model: str):
        """
        Initializes the retriever with a FAISS index, metadata, and an embedding model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = embed_model
        
        print(f"Loading model '{self.model_name}'...")
        if self.model_name.lower().startswith("cohere/"):
            if not COHERE_API_KEY or COHERE_API_KEY == "YOUR_COHERE_API_KEY_HERE":
                raise ValueError("Please provide a valid Cohere API key.")
            self.model = cohere.Client(COHERE_API_KEY)
        elif "jina-embeddings-v3" in self.model_name.lower():
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True, device=device)
        else:
            self.model = SentenceTransformer(self.model_name, device=device)

        print(f"Loading FAISS index from {faiss_path}...")
        self.index = faiss.read_index(faiss_path)
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.id2meta = {item["vector_id"]: item for item in metadata}
        print("Retriever initialized successfully.")


    def embed_query(self, query: str) -> np.ndarray:
        """
        Encodes a query into an embedding vector, handling different model types.
        """
        if isinstance(self.model, cohere.Client):
            print("Embedding query with Cohere API...")
            response = self.model.embed(
                texts=[query],
                model="embed-multilingual-v3.0",
                input_type="search_query" 
            )
            query_vec = response.embeddings
        else: # For all SentenceTransformer models
            print("Embedding query with SentenceTransformer...")
            query_vec = self.model.encode([query], normalize_embeddings=True)
            
        return np.array(query_vec, dtype="float32")

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Searches for the top_k nearest results in the FAISS index.
        """
        query_vec = self.embed_query(query)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            meta = self.id2meta.get(int(idx))
            if meta:
                results.append({
                    "id": int(idx),
                    "score": float(score),
                    "title": meta.get("title", "N/A"),
                    "chunk": meta.get("chunk", "N/A")
                })
        return results

# NEW FUNCTION: Ghi kết quả ra file JSON
def save_results_to_json(data: Dict, filename: str):
    """Saves the comparison results to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Comparison results have been saved to: {filename}")

# --- Main execution logic ---
def main():
    """
    Runs the retrieval comparison for all models and queries, then saves the result to a single JSON file.
    """
    # MODIFIED: Tạo một dictionary để lưu trữ tất cả kết quả
    final_results = {}

    for query in QUERIES_TO_TEST:
        print(f"\n==================================================")
        print(f"Executing tests for query: '{query}'")
        print(f"==================================================")
        
        # Cấu trúc kết quả cho từng truy vấn
        final_results[query] = {
            "results_by_model": {},
            "errors": []
        }

        for model_name in MODELS_TO_TEST:
            print(f"\n----- Processing model: {model_name} -----")
            safe_model_name = model_name.replace("/", "_")
            faiss_file_path = os.path.join(FAISS_DIR, f"{safe_model_name}.faiss")
            
            try:
                if not os.path.exists(faiss_file_path):
                    raise FileNotFoundError(f"FAISS file not found at {faiss_file_path}")
                if not os.path.exists(METADATA_PATH):
                     raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
                
                retriever = Retriever(
                    faiss_path=faiss_file_path,
                    metadata_path=METADATA_PATH,
                    embed_model=model_name
                )
                
                search_results = retriever.search(query, top_k=TOP_K)
                
                # Thêm rank vào kết quả và lưu trữ
                processed_results = []
                for rank, result in enumerate(search_results, 1):
                    result['rank'] = rank
                    processed_results.append(result)
                
                final_results[query]["results_by_model"][model_name] = processed_results

            except Exception as e:
                error_message = f"⚠️ Could not process model '{model_name}'. Error: {e}"
                final_results[query]["errors"].append(error_message)
                print(error_message)

    # MODIFIED: Ghi dictionary tổng hợp ra file JSON sau khi hoàn tất
    save_results_to_json(final_results, OUTPUT_JSON_FILE)

if __name__ == "__main__":
    main()