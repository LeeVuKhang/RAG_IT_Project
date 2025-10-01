import json
from chunking import Chunker
from deduplication import Deduplicator
from crawl import WebCrawler
from embedding import Embedder
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    # # ==========================================================
    # { -------------------- 1. Crawl -------------------- }
    # Crawl bài viết từ sitemap, tách chunk theo cấu trúc heading
    # ==========================================================
    sitemap_urls = [
        "https://dulichvn.net/post-sitemap1.xml",
        "https://dulichvn.net/post-sitemap2.xml",
        "https://dulichvn.net/post-sitemap3.xml",
        "https://dulichvn.net/post-sitemap4.xml",
    ]

    all_post_links = []
    for sm in sitemap_urls:
        all_post_links.extend(WebCrawler.get_post_links(sm))

    print(f" Tổng cộng {len(all_post_links)} link bài viết cần crawl")

    all_docs = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(WebCrawler.crawl_article, url): url for url in all_post_links}
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_docs.extend(result)

    with open("raw_crawl.json", "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print(f"Crawl xong, tổng cộng {len(all_docs)} chunks đã lưu!")

    # ==========================================================
    # { -------------------- 2. Chunking -------------------- }
    # Làm sạch dữ liệu, chunk lại và loại bỏ trùng lặp
    # ==========================================================
    input_file = "raw_crawl.json"
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tạo chunks từ dữ liệu crawl
    chunks = Chunker.make_chunks(data)

    # Loại bỏ duplicate bằng Deduplicator
    unique_chunks, added_hashes = Deduplicator.dedupe_chunks(chunks)

    # Ghi kết quả ra file JSON
    output_file = "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_chunks, f, ensure_ascii=False, indent=2)

    print(f"Đã xử lý {len(unique_chunks)} chunks (loại bỏ {len(chunks) - len(unique_chunks)} trùng lặp).")
    print(f"Kết quả lưu vào {output_file}")

    # ==========================================================
    # { -------------------- 3. Embedding -------------------- }
    # Sinh embeddings từ chunks, build FAISS index và metadata
    # ==========================================================
    # --- Setup ---
    device = Embedder.setup_device()
    model = Embedder.load_model(device)

    # --- Load dữ liệu ---
    file_path = "chunks.json"
    data = Embedder.load_chunks(file_path)

    # --- Chuẩn bị ---
    filtered_data, texts = Embedder.prepare_data(data)

    # --- Sinh embeddings ---
    embeddings = Embedder.build_embeddings(model, texts)

    # --- Build FAISS index ---
    Embedder.build_faiss_index(embeddings, faiss_file="embeddings.faiss")

    # --- Lưu metadata ---
    Embedder.save_metadata(filtered_data, metadata_file="metadata.json")

    print("pipeline: Crawl → Chunking → Deduplication → Embedding | Done!!!!")
