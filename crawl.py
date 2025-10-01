import requests
from bs4 import BeautifulSoup

class WebCrawler:
    @staticmethod
    def crawl_article(url):
        """
        Crawl nội dung 1 bài viết, gom theo H2.
        Bên trong H2: giữ nguyên H3 và P (mỗi cái cách nhau bằng \n).
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
            buffer = []

            def flush_buffer():
                nonlocal buffer, current_h2
                if buffer and current_h2:
                    content_text = "\n".join(buffer).strip()
                    if content_text:
                        contents.append({
                            "title": title.get_text(" ", strip=True),
                            "header": current_h2,
                            "content": content_text
                        })
                buffer = []

            for el in content_div.find_all(["h2", "h3", "p"]):
                if el.name == "h2":
                    flush_buffer()
                    current_h2 = el.get_text(" ", strip=True)
                elif el.name in ["h3", "p"]:
                    text = el.get_text(" ", strip=True)
                    if text:
                        buffer.append(text)  

            flush_buffer()  

            print(f"Crawl: {url} ({len(contents)} sections)")
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