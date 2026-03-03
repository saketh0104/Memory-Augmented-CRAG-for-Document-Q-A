from pydoc import text
from embeddings.embedder import Embedder
import numpy as np


class EvidenceQualityCritic:
    def __init__(self, threshold: float = 0.5, allow_soft_aggregation: bool = True):
        self.embedder = Embedder()
        self.threshold = threshold
        self.allow_soft_aggregation = allow_soft_aggregation

    @staticmethod
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def set_mode(self, threshold: float, allow_soft_aggregation: bool):
        self.threshold = threshold
        self.allow_soft_aggregation = allow_soft_aggregation

    def is_structural_noise(self, text: str) -> bool:
            noise_keywords = [
                "copyright",
                "isbn",
                "library of congress",
                "all rights reserved",
                "cover design",
                "title page",
                "published in",
                "imprint of",
                "lccn"
            ]
            
            t = text.lower()
            return any(k in t for k in noise_keywords)

    def evaluate(self, query: str, retrieved_chunks: list[dict]):
        """
        Returns:
          accepted_chunks: list
          needs_refinement: bool
        """


        query_emb = self.embedder.embed_query(query)

        accepted_chunks = []
        soft_chunks = []

        print("\n--- CRAG SCORES ---")
        for chunk in retrieved_chunks:

            if self.is_structural_noise(chunk["text"]):
                continue
             
            chunk_emb = self.embedder.embed_query(chunk["text"])
            score = self.cosine_similarity(query_emb, chunk_emb)
            print("Chunk", chunk["chunk_id"], "Score:", round(float(score), 3))
            
            if score >= self.threshold:
                chunk["relevance_score"] = round(float(score), 3)
                accepted_chunks.append(chunk)

            elif self.allow_soft_aggregation and score >= 0.35:
                chunk["relevance_score"] = round(float(score), 3)
                soft_chunks.append(chunk)

        # Decision logic
        if accepted_chunks:
            needs_refinement = False

        elif self.allow_soft_aggregation and len(soft_chunks) >= 2:
            accepted_chunks = soft_chunks[:3]
            needs_refinement = False

        else:
            needs_refinement = True

        return accepted_chunks, needs_refinement
