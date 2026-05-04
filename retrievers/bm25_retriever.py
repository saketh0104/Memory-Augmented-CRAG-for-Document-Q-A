from rank_bm25 import BM25Okapi
import re


class BM25Retriever:
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.tokenized_corpus = []
        self.bm25 = None

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def build_index(self, documents, metadatas):
        self.documents = documents
        self.metadatas = metadatas

        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=10):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "text": self.documents[idx],
                "source_file": self.metadatas[idx].get("source_file", "unknown"),
                "chunk_id": self.metadatas[idx].get("chunk_id", -1),
                "bm25_score": scores[idx]
            })

        return results