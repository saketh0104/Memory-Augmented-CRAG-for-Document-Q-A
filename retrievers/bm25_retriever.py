import os
import re
import pickle
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.tokenized_corpus = []
        self.bm25 = None

    # -------- TOKENIZER --------
    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    # -------- BUILD INDEX --------
    def build_index(self, documents, metadatas):
        if not documents:
            print("[BM25] No documents to index.")
            self.bm25 = None
            return

        self.documents = documents
        self.metadatas = metadatas

        self.tokenized_corpus = [
            self._tokenize(doc) for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"[BM25] Indexed {len(documents)} documents.")
        self.save_index()


    # ------- SAVE/LOAD INDEX --------
    def save_index(self, filepath="data/bm25_index.pkl"):
        os.makedirs("data", exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "documents": self.documents,
                "metadatas": self.metadatas
            }, f)

        print("[BM25] Index saved.")

    def load_index(self, filepath="data/bm25_index.pkl"):

        if not os.path.exists(filepath):
            return False

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]

        print("[BM25] Index loaded from disk.")
        return True
    

    # -------- SEARCH --------
    def search(self, query, top_k=10):
        if self.bm25 is None:
            print("[BM25] Index not built. Skipping BM25 retrieval.")
            return []

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        if len(scores) == 0:
            return []

        # -------- NORMALIZATION --------
        max_score = max(scores) if max(scores) > 0 else 1

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for idx in ranked_indices:
            try:
                results.append({
                    "text": self.documents[idx],
                    "source_file": self.metadatas[idx].get("source_file", "unknown"),
                    "chunk_id": self.metadatas[idx].get("chunk_id", -1),
                    "bm25_score": float(scores[idx]) / max_score  # ✅ normalized
                })
            except Exception:
                continue  # skip corrupted entries safely

        print(f"[BM25] Retrieved {len(results)} chunks.")

        return results