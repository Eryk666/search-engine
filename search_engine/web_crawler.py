import os
import json, hashlib
import asyncio, aiohttp, aiofiles
from bs4 import BeautifulSoup

from search_engine.page import Page


class WebCrawler:
    def __init__(
        self,
        seed_urls: list[str],
        max_depth: int = 1,
        scraped_urls_limit: int = 1000,
        output_dir: str = "data",
    ) -> None:
        self.session = None

        self.url_queue = asyncio.Queue()
        for seed_url in seed_urls:
            self.url_queue.put_nowait((seed_url, 0))
        self.seen_urls = set(seed_urls)

        self.max_depth = max_depth
        self.scraped_urls_count = 0
        self.scraped_urls_limit = scraped_urls_limit

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> None:
        asyncio.run(self.__crawl())
        print(f"Scraped {self.scraped_urls_count} urls!")

    async def __crawl(self) -> None:
        try:
            self.session = aiohttp.ClientSession()
            tasks = [asyncio.create_task(self.__crawl_task()) for _ in range(8)]
            await asyncio.gather(*tasks)
        finally:
            await self.session.close()

    async def __crawl_task(self) -> None:
        while self.scraped_urls_count < self.scraped_urls_limit:
            try:
                url, depth = await asyncio.wait_for(self.url_queue.get(), timeout=3.0)
                async with self.session.get(url, timeout=5.0) as response:
                    html = await response.text("utf-8", "ignore")
                    soup = BeautifulSoup(html, "html.parser")
                    page = Page(url, soup)
                    await self.__save_page_to_json(page)
                    await self.__enqueue_page_urls(page, depth)
            except asyncio.TimeoutError:
                break
            except (aiohttp.ClientConnectionError, TimeoutError):
                continue
            finally:
                self.url_queue.task_done()

            self.scraped_urls_count += 1

    async def __enqueue_page_urls(self, page: Page, page_depth: int) -> None:
        if page_depth >= self.max_depth:
            return

        for url in page.get_forward_urls():
            if url not in self.seen_urls and page.is_same_domain(url):
                self.seen_urls.add(url)
                await self.url_queue.put((url, page_depth + 1))

    async def __save_page_to_json(self, page: Page) -> None:
        filename = self.__get_filename_from_url(page.url)
        file_path = os.path.join(self.output_dir, filename)

        page_data = {
            "url": page.url,
            "title": page.get_title(),
            "description": page.get_description(),
            "words": page.get_words(),
        }

        async with aiofiles.open(file_path, "w") as file:
            await file.write(json.dumps(page_data, indent=4))

    def __get_filename_from_url(self, url: str) -> str:
        return hashlib.md5(url.encode("utf-8")).hexdigest() + ".json"
