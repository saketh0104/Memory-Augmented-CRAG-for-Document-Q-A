from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(
        self,
        model_path="models/ms-marco-MiniLM-L-6-v2"
    ):
        print("[RERANKER] Loading cross encoder...")

        self.model = CrossEncoder(model_path)

        print("[RERANKER] Ready.")

    def rerank(
        self,
        query,
        retrieved_chunks,
        top_k=5
    ):

        if not retrieved_chunks:
            return []

        pairs = [
            [query, chunk["text"]]
            for chunk in retrieved_chunks
        ]

        scores = self.model.predict(pairs)

        for chunk, score in zip(retrieved_chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(
            retrieved_chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]