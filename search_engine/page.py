from urllib.parse import urljoin, urlparse
from nltk.tokenize import word_tokenize
from search_engine.util import normalize_text, is_valid_word


class Page:
    def __init__(self, url, soup):
        self.url = url
        self.soup = soup

    def get_title(self):
        title_tag = self.soup.title
        if title_tag:
            title = self.soup.title.text
            return normalize_text(title)

        og_title_tag = self.soup.find("meta", attrs={"property": "og:title"})
        if og_title_tag:
            title = og_title_tag.get("content")
            return normalize_text(title)

        twitter_title_tag = self.soup.find("meta", attr={"name": "twitter:title"})
        if twitter_title_tag:
            title = twitter_title_tag.get("content")
            return normalize_text(title)

        return ""

    def get_description(self):
        description_tag = self.soup.find("meta", attrs={"name": "description"})
        if description_tag:
            description = description_tag.get("content")
            return normalize_text(description)

        og_description_tag = self.soup.find(
            "meta", attrs={"property": "og:description"}
        )
        if og_description_tag:
            description = og_description_tag.get("content")
            return normalize_text(description)

        twitter_description_tag = self.soup.find(
            "meta", attrs={"name": "twitter:description"}
        )
        if twitter_description_tag:
            description = twitter_description_tag.get("content")
            return normalize_text(description)

        return ""

    def get_words(self):
        words = {}

        for word in word_tokenize(self.soup.get_text(" ")):
            word = normalize_text(word.casefold())
            if is_valid_word(word):
                words[word] = words.get(word, 0) + 1

        return words

    def get_forward_urls(self):
        urls = []
        a_tags = self.soup.find_all("a", href=True)

        for a_tag in a_tags:
            urls.append(urljoin(self.url, a_tag["href"]))

        return urls

    def is_same_domain(self, other_url):
        parsed_url = urlparse(self.url)
        other_parsed_url = urlparse(other_url)

        return (
            other_parsed_url.scheme == parsed_url.scheme
            and other_parsed_url.netloc == parsed_url.netloc
        )