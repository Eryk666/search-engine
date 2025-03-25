import os
import json
import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.linalg import svds
from typing import List, Dict, Tuple


class SearchMatrix:
    """
    Stores word frequencies in a sparse matrix and its low-rank approximation.

    Attributes:
        words (List[str]): List of unique words.
        pages (List[Tuple[str, str, str]]): List of (URL, Title, Description) tuples.
        word_frequency (spmatrix): Sparse matrix of word frequencies.
        word_frequency_low_rank (spmatrix): Low-rank approximation of word_frequency.
        word_to_index (Dict[str, int]): Mapping from word to its index in words.
        page_to_index (Dict[str, int]): Mapping from URL to its index in pages.
    """

    def __init__(
        self,
        words: List[str],
        pages: List[Tuple[str, str, str]],
        word_frequency: spmatrix,
        svd_rank: int,
    ):
        """
        Initializes the SearchMatrix and computes the low-rank approximation.

        Args:
            words (List[str]): List of unique words.
            pages (List[Tuple[str, str, str]]): List of (URL, Title, Description) tuples.
            word_frequency (spmatrix): Sparse matrix of word frequencies.
            svd_rank (int): Rank for SVD approximation.
        """
        self.words = words
        self.pages = pages
        self.svd_rank = svd_rank

        # Reverse mappings for fast lookup
        self.word_to_index = {word: i for i, word in enumerate(words)}
        self.page_to_index = {url: i for i, (url, _, _) in enumerate(pages)}

        self.word_frequency = word_frequency
        self.word_frequency_low_rank = self.__compute_svd(svd_rank)

    def __compute_svd(self, rank: int) -> spmatrix:
        """
        Computes a low-rank approximation of the word frequency matrix using Singular Value Decomposition (SVD).

        Args:
            rank (int): The number of singular values to keep.

        Returns:
            spmatrix: The low-rank approximation stored as a sparse matrix.

        Raises:
            ValueError: If the rank is too large.
        """
        if rank >= min(self.word_frequency.shape):
            raise ValueError("Rank must be smaller than the smallest matrix dimension.")

        U, Sigma, Vt = svds(self.word_frequency.astype(np.float32), k=rank)

        return csr_matrix(U @ np.diag(Sigma) @ Vt)

    def __repr__(self) -> str:
        return (
            f"SearchMatrix(words_count={len(self.words)}, "
            f"pages_count={len(self.pages)}, "
            f"svd_rank={self.svd_rank})"
        )


def load_search_matrix(folder_path: str, svd_rank: int) -> SearchMatrix:
    """
    Loads JSON files from a folder and constructs a SearchMatrix.

    Args:
        folder_path (str): Path to the folder containing JSON files.
        svd_rank (int): Rank for SVD approximation.

    Returns:
        SearchMatrix: The constructed SearchMatrix instance.
    """

    words: List[str] = []
    pages: List[Tuple[str, str, str]] = []  # [(URL, Title, Description)]
    word_to_index: Dict[str, int] = {}
    page_to_index: Dict[str, int] = {}
    word_counts: Dict[int, Dict[int, int]] = {}  # {page_index: {word_index: count}}

    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), "r") as f:
            data = json.load(f)
            url: str = data["url"]
            title: str = data["title"]
            description: str = data["description"]
            word_frequencies: Dict[str, int] = data["words"]

            page_index = len(pages)
            page_to_index[url] = page_index
            pages.append((url, title, description))

            word_counts[page_index] = {}

            for word, count in word_frequencies.items():
                if word not in word_to_index:
                    word_to_index[word] = len(words)
                    words.append(word)

                word_index = word_to_index[word]
                word_counts[page_index][word_index] = count

    rows, cols, data = [], [], []
    for page_index, page_word_counts in word_counts.items():
        for word_index, count in page_word_counts.items():
            rows.append(word_index)
            cols.append(page_index)
            data.append(count)

    word_frequency = csr_matrix((data, (rows, cols)))

    return SearchMatrix(words, pages, word_frequency, svd_rank)
