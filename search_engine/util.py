import re, string
from nltk.corpus import stopwords


stop_words = set(stopwords.words("english")) | set(string.punctuation)


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()
    

def is_valid_word(word):
    if word and bool(re.match(r'^[a-zA-Z]+$', word)) and word not in stop_words:
        return True
    return False