import json
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import os
import re
import cohere

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Please replace with your actual API key or load it from environment variables
COHERE_API_KEY = "ws2oGGyOm5ewVRwiXqiLiugOzpkdckTwbu7Bo18w"

MODELS_TO_TEST = [
    "AITeamVN/Vietnamese_Embedding",
    "BAAI/bge-m3",
    "Cohere/embed-multilingual-v3.0",
    "jinaai/jina-embeddings-v3"
]

INPUT_FILE = "data.json"
OUTPUT_DIR = "Testing"
FAISS_DIR = os.path.join(OUTPUT_DIR, "faiss")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
RESULTS_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "results_summary.json")

class Embedder:
    """
    Utility class for handling the creation and management of text embeddings.
    All methods are static.
    """
    @staticmethod
    def setup_device():
        """Sets up the device (CUDA if available, otherwise CPU)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        return device

    @staticmethod
    def load_model(device, model_name):
        """Loads an embedding model based on its name."""
        print(f"Loading model: {model_name}...")
        if model_name.lower().startswith("cohere/"):
            if not COHERE_API_KEY:
                raise ValueError("Please provide a valid Cohere API key.")
            return cohere.Client(COHERE_API_KEY)
        elif "jina-embeddings-v3" in model_name.lower():
            return SentenceTransformer(model_name, trust_remote_code=True, device=device)
        else:
            return SentenceTransformer(model_name, device=device)

    @staticmethod
    def load_chunks(file_path):
        """Loads text data from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def prepare_data(data):
        """Prepares and cleans the data, creating the text format for embedding."""
        processed_data = []
        for idx, item in enumerate(data):
            text = item.get("text") or item.get("chunk", "")
            text_to_embed = f"Title: {item.get('title','')} - Section: {item.get('header','')}\nContent: {text}"
            
            item_copy = item.copy()
            item_copy["chunk"] = text
            item_copy["text_to_embed"] = text_to_embed
            item_copy["vector_id"] = idx
            processed_data.append(item_copy)
            
        texts_to_embed = [item["text_to_embed"] for item in processed_data]
        print(f"Prepared {len(processed_data)} data chunks.")
        return processed_data, texts_to_embed

    @staticmethod
    def build_embeddings(model, texts, model_name, batch_size=32):
        """Creates embeddings from text, with specific handling for different model types."""
        print("Starting the embedding process...")
        if isinstance(model, cohere.Client):
            response = model.embed(
                texts=texts,
                model="embed-multilingual-v3.0",
                input_type="search_document"
            )
            embeddings = np.array(response.embeddings, dtype=np.float32)
        else: # For SentenceTransformer models
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True
            )
        print("Embeddings created successfully.")
        return embeddings

    @staticmethod
    def build_faiss_index(embeddings, faiss_file):
        """Builds and saves a FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) 
        index.add(np.asarray(embeddings, dtype="float32"))
        
        faiss.write_index(index, faiss_file)
        print(f"FAISS index saved to: {faiss_file}")
        print(f"  -> Vectors: {index.ntotal}, Dimensions: {dimension}")
        return index, dimension

    @staticmethod
    def save_metadata(processed_data, metadata_file):
        """Saves metadata related to the text chunks."""
        metadata = [
            {
                "vector_id": item["vector_id"],
                "id": item.get("id"),
                "title": item.get("title"),
                "header": item.get("header"),
                "chunk": item.get("chunk"),
            }
            for item in processed_data
        ]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        print(f"Metadata saved to: {metadata_file}")

def run_experiment(model_name, device):
    """Executes the entire workflow for a single model."""
    print("\n" + "="*60)
    print(f"Processing model: {model_name}")
    print("="*60)
    
    report = {"model_name": model_name}
    sanitized_name = re.sub(r'[/]', '_', model_name)
    start_time = time.time()

    try:
        model = Embedder.load_model(device, model_name)
        data = Embedder.load_chunks(INPUT_FILE)
        processed_data, texts_to_embed = Embedder.prepare_data(data)
        embeddings = Embedder.build_embeddings(model, texts_to_embed, model_name)
        
        faiss_file = os.path.join(FAISS_DIR, f"{sanitized_name}.faiss")
        metadata_file = os.path.join(METADATA_DIR, f"{sanitized_name}.json")
        
        index, dim = Embedder.build_faiss_index(embeddings, faiss_file)
        Embedder.save_metadata(processed_data, metadata_file)

        total_time = time.time() - start_time
        report.update({
            "status": "Success",
            "embedding_shape": embeddings.shape,
            "processing_time_s": round(total_time, 2),
            "num_chunks": len(texts_to_embed),
            "dimension": dim,
            "faiss_file": faiss_file,
            "metadata_file": metadata_file,
            "error": None
        })
        print(f"Finished processing {model_name} in {total_time:.2f} seconds.")

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        report.update({
            "status": "Failure",
            "error": error_message,
            "processing_time_s": round(time.time() - start_time, 2)
        })
        print(f"Error with model '{model_name}': {error_message}")
        print("  -> Skipping to the next model.")
        
    return report

if __name__ == "__main__":
    # Create sample data
    test_data = [
        {"id": "chunk_0", "title": "Travel Guide", "header": "Introduction", "chunk": "This is a wonderful destination with many islands to explore."},
        {"id": "chunk_1", "title": "Travel Guide", "header": "Best Time to Visit", "chunk": "The ideal time to travel is from December to March each year."},
        {"id": "chunk_2", "title": "Local Cuisine", "header": "Seafood", "chunk": "The locals are mainly fishermen, so the seafood is always fresh and reasonably priced."}
    ]
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    print(f"Sample data file created: {INPUT_FILE}")

    # Ensure output directories exist
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Run experiments
    device = Embedder.setup_device()
    results = [run_experiment(model_name, device) for model_name in MODELS_TO_TEST]

    # Save results to a file
    with open(RESULTS_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\nSummary results saved to: {RESULTS_SUMMARY_FILE}")