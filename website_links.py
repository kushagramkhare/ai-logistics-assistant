import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

BASE_URL = "https://iitj.ac.in/"
visited_links = set()   # stores all discovered links
pdf_links = set() #stores pdfs


def is_internal(url):
    """Check if link belongs to same website"""
    return urlparse(url).netloc == urlparse(BASE_URL).netloc

def is_english(url):
    url = url.lower()

    # If Hindi indicators exist → reject
    if "/hi/" in url:
        return False
    if url.endswith("/hi"):
        return False
    if "lang=hi" in url:
        return False
    if "lg=hi" in url:
        return False
    if url.endswith("hi"):
        return False
    
    

    # Otherwise allow
    return True


def dfs(url):
    if url in visited_links:
        return
    if url.lower().endswith(".pdf"):
        pdf_links.add(url)
        print("Visiting:", url)
        return

    
    visited_links.add(url)

    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        # find all links on the page
        for tag in soup.find_all("a", href=True):
            link = urljoin(url, tag["href"]).split("#")[0]
            if not is_internal(link) or not is_english(link):
                continue

            if link.lower().endswith(".pdf"):
                pdf_links.add(link)
                print("Visiting:", link)
            else:
                dfs(link)  # DFS only for webpages

        time.sleep(1)  # be polite to server

    except:
        pass


# Start DFS from homepage
dfs(BASE_URL)

print("\nTotal links found:", len(visited_links))
