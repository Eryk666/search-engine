import os
import json
import hashlib
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from nltk.tokenize import word_tokenize
from search_engine.util import normalize_text, is_valid_word


class WebCrawler:
    """
    A WebCrawler that crawls seed URLs, extracts data (title, description, word frequency), and saves it to JSON files.
    """

    def __init__(
        self,
        seed_urls: list[str],
        max_depth: int = 1,
        scraped_urls_limit: int = 1000,
        output_dir: str = "data",
    ) -> None:
        self.session: aiohttp.ClientSession | None = None
        self.url_queue: asyncio.Queue = asyncio.Queue()
        for seed_url in seed_urls:
            self.url_queue.put_nowait((seed_url, 0))
        self.seen_urls: set[str] = set(seed_urls)
        self.max_depth: int = max_depth
        self.scraped_urls_count: int = 0
        self.scraped_urls_limit: int = scraped_urls_limit
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> None:
        """Start the crawling process."""
        asyncio.run(self.__crawl())
        print(f"Scraped {self.scraped_urls_count} urls!")

    async def __crawl(self) -> None:
        """Main crawling task that manages multiple concurrent crawl tasks."""
        try:
            self.session = aiohttp.ClientSession()
            tasks = [asyncio.create_task(self.__crawl_task()) for _ in range(8)]
            await asyncio.gather(*tasks)
        finally:
            await self.session.close()

    async def __crawl_task(self) -> None:
        """Handle crawling a page, extracting info, and enqueuing new URLs."""
        while self.scraped_urls_count < self.scraped_urls_limit:
            try:
                url, depth = await asyncio.wait_for(self.url_queue.get(), timeout=3.0)
                async with self.session.get(url, timeout=5.0) as response:
                    html = await response.text("utf-8", "ignore")
                    soup = BeautifulSoup(html, "html.parser")
                    await self.__save_page_to_json(url, soup)
                    await self.__enqueue_page_urls(url, soup, depth)
            except asyncio.TimeoutError:
                break
            except (aiohttp.ClientConnectionError, TimeoutError):
                continue
            finally:
                self.url_queue.task_done()
            self.scraped_urls_count += 1

    async def __enqueue_page_urls(
        self, url: str, soup: BeautifulSoup, page_depth: int
    ) -> None:
        """Enqueue URLs found on the page if they haven’t been seen and are within the same domain."""
        if page_depth >= self.max_depth:
            return
        for new_url in self.__get_forward_urls(url, soup):
            if new_url not in self.seen_urls and self.__is_same_domain(url, new_url):
                self.seen_urls.add(new_url)
                await self.url_queue.put((new_url, page_depth + 1))

    async def __save_page_to_json(self, url: str, soup: BeautifulSoup) -> None:
        """Save page data (URL, title, description, word frequency) to a JSON file."""
        filename = self.__get_filename_from_url(url)
        file_path = os.path.join(self.output_dir, filename)

        page_data = {
            "url": url,
            "title": self.__get_title(soup),
            "description": self.__get_description(soup),
            "words": self.__get_words(soup),
        }

        async with aiofiles.open(file_path, "w") as file:
            await file.write(json.dumps(page_data, indent=4))

    def __get_filename_from_url(self, url: str) -> str:
        """Generate a filename for page data based on its URL using MD5 hash."""
        return hashlib.md5(url.encode("utf-8")).hexdigest() + ".json"

    def __get_title(self, soup: BeautifulSoup) -> str:
        """Extract the title of the page from its HTML or meta tags."""
        title_tag = soup.title
        if title_tag:
            title = soup.title.text
            return normalize_text(title)

        og_title_tag = soup.find("meta", attrs={"property": "og:title"})
        if og_title_tag:
            title = og_title_tag.get("content")
            return normalize_text(title)

        twitter_title_tag = soup.find("meta", attrs={"name": "twitter:title"})
        if twitter_title_tag:
            title = twitter_title_tag.get("content")
            return normalize_text(title)

        return ""

    def __get_description(self, soup: BeautifulSoup) -> str:
        """Extract the description of the page from its HTML or meta tags."""
        description_tag = soup.find("meta", attrs={"name": "description"})
        if description_tag:
            description = description_tag.get("content")
            return normalize_text(description)

        og_description_tag = soup.find("meta", attrs={"property": "og:description"})
        if og_description_tag:
            description = og_description_tag.get("content")
            return normalize_text(description)

        twitter_description_tag = soup.find(
            "meta", attrs={"name": "twitter:description"}
        )
        if twitter_description_tag:
            description = twitter_description_tag.get("content")
            return normalize_text(description)

        return ""

    def __get_words(self, soup: BeautifulSoup) -> dict[str, int]:
        """Extract word frequencies from the page text."""
        words = {}
        for word in word_tokenize(soup.get_text(" ")):
            word = normalize_text(word.casefold())
            if is_valid_word(word):
                words[word] = words.get(word, 0) + 1
        return words

    def __get_forward_urls(self, url: str, soup: BeautifulSoup) -> list[str]:
        """Extract all forward links (URLs) from the page."""
        urls = []
        a_tags = soup.find_all("a", href=True)
        for a_tag in a_tags:
            urls.append(urljoin(url, a_tag["href"]))
        return urls

    def __is_same_domain(self, url: str, other_url: str) -> bool:
        """Check if the other URL belongs to the same domain."""
        parsed_url = urlparse(url)
        other_parsed_url = urlparse(other_url)
        return (
            other_parsed_url.scheme == parsed_url.scheme
            and other_parsed_url.netloc == parsed_url.netloc
        )
