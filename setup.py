import nltk
from config import SEED_URLS, MAX_DEPTH, SCRAPED_URLS_LIMIT, DATA_DIR
from search_engine.web_crawler import WebCrawler


if __name__ == "__main__":
    # Download necessary NLTK data
    nltk.download("stopwords")
    nltk.download("punkt")

    # Run the web crawler
    crawler = WebCrawler(SEED_URLS, MAX_DEPTH, SCRAPED_URLS_LIMIT, DATA_DIR)
    crawler.run()

    print("Web crawling is complete. You can now start the app.")
