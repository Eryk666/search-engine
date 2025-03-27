# Search Engine

A simple search engine application that retrieves relevant articles based on user queries. It uses a custom
web crawler to gather the articles and leverages SVD for efficient and accurate search results. Made for
the Computational Methods in Science and Technology course at AGH UST.

## Installation

You'll need Python 3.8 or higher and Poetry installed. Then simply run the following command:

```bash
poetry install
```

## Usage

Before using the search engine, you need to collect the documents by running the setup script. This process involves crawling web pages and depending on the number of pages being scraped, it may take a while.

To collect the documents:

```bash
python setup.py
```

Once the setup is complete, you can run the Flask server and start using the search engine:

```bash
python app.py
```

## Configuration

You can configure various aspects of the search engine through the config.py file. Here’s what each configuration option does:

- `DATA_DIR`: Specifies the directory where the collected data (web pages) will be stored.
- `RANK`: The rank for SVD approximation. This determines the number of dimensions to reduce the original matrix to. A higher value will increase accuracy but also computational cost.
- `USE_IDF`: Whether to use Inverse Document Frequency (IDF) weighting in the search engine. Can improve the relevance of search results by emphasizing rarer terms.
- `SCRAPED_URLS_LIMIT`: Defines the number of URLs that the web crawler will scrape.
- `SEED_URLS`: A list of initial URLs for the web crawler to start scraping.
- `MAX_DEPTH`: Specifies the maximum depth for crawling the websites.
