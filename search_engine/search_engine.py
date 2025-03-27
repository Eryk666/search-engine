from typing import List, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from search_engine.search_matrix import SearchMatrix


class SearchEngine:
    """
    A search engine that retrieves relevant documents based on a query.

    Attributes:
        search_matrix (SearchMatrix): The underlying search matrix used for query matching.
    """

    def __init__(self, search_matrix: SearchMatrix) -> None:
        """
        Initializes the SearchEngine with a search matrix.

        Args:
            search_matrix (SearchMatrix): The search matrix used to store and retrieve documents.
        """
        self.search_matrix = search_matrix

    def search(
        self, query: str, results_count: int, filter_noise: bool = True
    ) -> List[Tuple[str, str, str, float]]:
        """
        Searches for the most relevant documents matching the query.

        Args:
            query (str): The search query.
            results_count (int): Number of results to return.
            filter_noise (bool): Whether to apply noise filtering for better accuracy.

        Returns:
            (List[Tuple[str, str, str, float]]): A list of (document_id, title, snippet, relevance_score).
        """
        query_vector = self.get_query_vector(query)
        match_scores = self.get_match_scores(query_vector, filter_noise)
        n_best_results = self.get_top_results(match_scores, results_count)

        return n_best_results

    def get_query_vector(self, query: str) -> csr_matrix:
        """
        Converts a query string into a normalized sparse query vector.

        Args:
            query (str): The search query.

        Returns:
            csr_matrix: A sparse, normalized query vector matching the word order in search_matrix.
        """
        query_words = query.lower().split()
        word_counts = {}

        for word in query_words:
            if word in self.search_matrix.word_to_index:
                word_counts[word] = word_counts.get(word, 0) + 1

        if not word_counts:
            return csr_matrix((len(self.search_matrix.words), 1), dtype=np.float32)

        indices = [
            self.search_matrix.word_to_index[word] for word in word_counts.keys()
        ]
        values = list(word_counts.values())
        query_vector = csr_matrix(
            (values, (indices, np.zeros(len(indices)))),
            shape=(len(self.search_matrix.words), 1),
            dtype=np.float32,
        )

        return normalize(query_vector, axis=0)

    def get_match_scores(
        self, query_vector: csr_matrix, filter_noise: bool
    ) -> np.ndarray:
        """
        Computes match scores between the query vector and search matrix.

        Args:
            query_vector (csr_matrix): The normalized sparse query vector.
            filter_noise (bool): Whether to use the low-rank approximation for denoising.

        Returns:
            np.ndarray: A 1D NumPy array of relevance scores for each document.
        """
        word_frequency = None

        if filter_noise:
            word_frequency = self.search_matrix.word_frequency_low_rank
        else:
            word_frequency = self.search_matrix.word_frequency

        scores = np.abs((query_vector.T @ word_frequency).toarray().flatten())
        scores = scores / np.linalg.norm(scores)

        return scores

    def get_top_results(
        self, match_scores: np.ndarray, results_count: int
    ) -> List[Tuple[str, str, str, float]]:
        """
        Returns the documents with the highest match scores.

        Args:
            match_scores (np.ndarray): An array of relevance scores for each document.
            results_count (int): The number of top results to return.

        Returns:
            List[Tuple[str, str, str, float]]: A list of (URL, title, description, match_score),
            sorted by match score in descending order.
        """
        if results_count <= 0:
            return []

        top_indices = np.argsort(match_scores)[::-1][:results_count]

        top_results = [
            (
                self.search_matrix.pages[idx][0],  # URL
                self.search_matrix.pages[idx][1],  # Title
                self.search_matrix.pages[idx][2],  # Description
                round(float(match_scores[idx]), 2),  # Match score as float
            )
            for idx in top_indices
        ]

        return top_results


if __name__ == "__main__":
    search_engine = SearchEngine()
    print(search_engine.term_by_page_matrix.shape)
    print(search_engine.term_by_page_matrix_approx.shape)
