import requests
from bs4 import BeautifulSoup

class WebCrawler:
    @staticmethod
    def crawl_article(url):
        """
        Crawl nội dung 1 bài viết, gom theo heading (h2, h3) và đoạn văn (p).
        """
        try:
            res = requests.get(url, timeout=10)
            res.encoding = "utf-8"
            soup = BeautifulSoup(res.text, "html.parser")

            title = soup.select_one("h1.entry-title")
            content_div = soup.select_one("div.entry-content")

            if not title or not content_div:
                print(f"Bỏ qua (không có nội dung bài viết): {url}")
                return []

            contents = []
            current_h2 = None
            current_h3 = None
            buffer = []

            def flush_buffer():
                nonlocal buffer, current_h2, current_h3
                if buffer:
                    header = current_h3 if current_h3 else current_h2
                    content_text = "\n".join(buffer).strip()
                    if content_text:
                        content_data = {
                            "title": title.get_text(" ", strip=True),
                            "content": content_text
                        }
                        if header:
                            content_data["header"] = header
                        contents.append(content_data)
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

            flush_buffer()  

            print(f"Crawl: {url} ({len(contents)} contents)")
            return contents

        except Exception as e:
            print(f"Lỗi khi crawl {url}: {e}")
            return []

    @staticmethod
    def get_post_links(sitemap_url):
        """Lấy tất cả link bài viết từ 1 sitemap"""
        try:
            resp = requests.get(sitemap_url, timeout=10)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.content, "xml")
            all_links = [loc.text for loc in soup.find_all("loc")]
            post_links = [link for link in all_links if "/wp-content/" not in link]
            print(f"{sitemap_url}: lấy {len(post_links)} link")
            return post_links
        except Exception as e:
            print(f"Lỗi khi lấy link từ {sitemap_url}: {e}")
            return []