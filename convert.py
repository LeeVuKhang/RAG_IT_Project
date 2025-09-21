import json

# Đọc file dulichvietnam.com.vn
with open("dulichvn_sitemap1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

converted = []
for article in data:
    url = article["url"]
    title = article["title"]
    for sec in article.get("sections", []):
        content = sec.get("content", "").strip()
        if content:
            converted.append({
                "url": url,
                "title": title,
                "chunk": content
            })

# Lưu ra file mới
with open("dulichvn_chunks.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print(f"Đã chuyển {len(converted)} chunks")