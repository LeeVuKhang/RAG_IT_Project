import requests
from bs4 import BeautifulSoup
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def recursive_split(text, max_words=500):
    """Chia nhỏ text dài thành nhiều đoạn nhỏ (Level 2)"""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end
    return chunks


def crawl_article(url):
    """Crawl nội dung 1 bài viết và chia theo cấu trúc heading (Level 3 + fallback Level 2)"""
    try:
        res = requests.get(url, timeout=10)
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")

        title = soup.select_one("h1.entry-title")
        content_div = soup.select_one("div.entry-content")

        if not title or not content_div:
            print(f"⚠️ Bỏ qua (không có nội dung bài viết): {url}")
            return []

        chunks = []
        current_h2 = None
        current_h3 = None
        buffer = []

        def flush_buffer():
            """Lưu nội dung buffer thành chunk (có fallback Level 2)"""
            nonlocal buffer, current_h2, current_h3
            if buffer:
                header = current_h3 if current_h3 else current_h2
                content_text = "\n".join(buffer).strip()
                if content_text:
                    # Nếu quá dài thì fallback sang recursive split
                    if len(content_text.split()) > 1000:
                        split_contents = recursive_split(content_text, max_words=500)
                        for idx, sub_content in enumerate(split_contents, start=1):
                            chunks.append({
                                "url": url,
                                "title": title.get_text(" ", strip=True),
                                "header": f"{header} (phần {idx})" if header else None,
                                "chunk": f"{title.get_text(' ', strip=True)}\n{header} (phần {idx})\n{sub_content}"
                            })
                    else:
                        chunks.append({
                            "url": url,
                            "title": title.get_text(" ", strip=True),
                            "header": header,
                            "chunk": f"{title.get_text(' ', strip=True)}\n{header}\n{content_text}"
                        })
                buffer = []

        for el in content_div.find_all(["h2", "h3", "p"]):
            if el.name == "h2":
                flush_buffer()
                current_h2 = el.get_text(" ", strip=True)
                current_h3 = None
            elif el.name == "h3":
                flush_buffer()
                current_h3 = el.get_text(" ", strip=True)
            elif el.name == "p":
                text = el.get_text(" ", strip=True)
                if text:
                    buffer.append(text)

        flush_buffer()  # xử lý đoạn cuối

        print(f"✅ Đã crawl & split (Level 3 + fallback 2): {url} ({len(chunks)} chunks)")
        return chunks

    except Exception as e:
        print(f"❌ Lỗi khi crawl {url}: {e}")
        return []


def get_post_links(sitemap_url):
    """Lấy tất cả link bài viết từ 1 sitemap"""
    try:
        resp = requests.get(sitemap_url, timeout=10)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.content, "xml")
        all_links = [loc.text for loc in soup.find_all("loc")]
        post_links = [link for link in all_links if "/wp-content/" not in link]
        print(f"🔎 {sitemap_url}: lấy {len(post_links)} link")
        return post_links
    except Exception as e:
        print(f"❌ Lỗi khi lấy link từ {sitemap_url}: {e}")
        return []


def main():
    sitemap_urls = [
        "https://dulichvn.net/post-sitemap1.xml",
        "https://dulichvn.net/post-sitemap2.xml",
        "https://dulichvn.net/post-sitemap3.xml",
        "https://dulichvn.net/post-sitemap4.xml",
    ]

    all_post_links = []
    for sm in sitemap_urls:
        all_post_links.extend(get_post_links(sm))

    print(f"📌 Tổng cộng {len(all_post_links)} link bài viết cần crawl")

    all_chunks = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(crawl_article, url): url for url in all_post_links}
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_chunks.extend(result)

    with open("dulichvn_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"📂 Crawl + split (Level 3 + fallback 2) xong, tổng cộng {len(all_chunks)} chunks đã lưu!")


if __name__ == "__main__":
    main()
