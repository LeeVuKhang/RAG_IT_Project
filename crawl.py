import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time

BASE_URL = "https://dulichvietnam.com.vn/cam-nang-trong-nuoc.html"
DOMAIN = "https://dulichvietnam.com.vn"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/116.0.0.0 Safari/537.36"
}

SKIP_TITLES = {"Trang chủ", "Việt Nam"}

def get_page(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text

def parse_list_page(html):
    soup = BeautifulSoup(html, "html.parser")
    articles = []
    for a in soup.select(".vnt-main a[href]"):
        href = a["href"].strip()
        title = a.get_text(strip=True)

        if not title or title in SKIP_TITLES:
            continue
        if href.startswith("#") or "javascript" in href.lower():
            continue
        if "page=" in href.lower():
            continue
        if href.startswith("/cam-nang-trong-nuoc"):
            continue

        # Chỉ lấy link bài viết thật
        if href.startswith("/"):
            full_url = urljoin(DOMAIN, href)
            articles.append({"url": full_url, "title": title})
    return articles

def parse_detail_page(url, title):
    html = get_page(url)
    soup = BeautifulSoup(html, "html.parser")
    sections = []
    main_div = soup.select_one(".vnt-main")
    if not main_div:
        return {"url": url, "title": title, "sections": []}

    current_header = ""
    for tag in main_div.find_all(["h1", "h2", "h3", "p"], recursive=True):
        if tag.name in ["h1", "h2", "h3"]:
            current_header = tag.get_text(strip=True)
        elif tag.name == "p":
            content = tag.get_text(strip=True)
            if content:
                sections.append({"header": current_header, "content": content})
    return {"url": url, "title": title, "sections": sections}

def crawl_all_pages(max_pages=8):
    all_data = []
    for page_num in range(1, max_pages + 1):
        page_url = BASE_URL if page_num == 1 else f"{BASE_URL}?page={page_num}"
        print(f"Đang lấy: {page_url}")
        html = get_page(page_url)
        articles = parse_list_page(html)
        if not articles:
            break
        for art in articles:
            print(f"  -> Lấy bài: {art['title']}")
            detail = parse_detail_page(art["url"], art["title"])
            all_data.append(detail)
            time.sleep(1)  
    return all_data

if __name__ == "__main__":
    data = crawl_all_pages()
    with open("dulichvn_sitemap1.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(data)} bài vào dulichvn_sitemap1.json")