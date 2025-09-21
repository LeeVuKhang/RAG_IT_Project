import json
import glob
import pandas as pd

all_chunks = []
for file in glob.glob("dulichvn_sitemap.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        url = item.get("url", "")
        title = item.get("title", "")

        for sec in item.get("sections", []):
            header = sec.get("header") or ""
            content = sec.get("content") or ""
            text = (header + " " + content).strip()


            if text:
                all_chunks.append({
                    "url": url,
                    "title": title,
                    "chunk": text
                })

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(all_chunks)
df.to_csv("chunks.csv", index=False, encoding="utf-8-sig")

print(f"✅ Đã tạo {len(all_chunks)} chunks, lưu vào chunks.json và chunks.csv")
