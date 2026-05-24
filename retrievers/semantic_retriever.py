from sklearn.metrics.pairwise import cosine_similarity


class SemanticRetriever:
    def __init__(self, vector_db, embedder):
        self.vector_db = vector_db
        self.embedder = embedder

    def search(self, query, top_k=10):
        query_embedding = self.embedder.embed_query(query)

        results = self.vector_db.query_documents(
            query_embedding=query_embedding,
            top_k=top_k
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        semantic_chunks = []

        for text, meta in zip(docs, metas):

            if not text.strip():
                continue

            # similarity scoring
            doc_embedding = self.embedder.embed_query(text)

            sim = cosine_similarity(
                [query_embedding],
                [doc_embedding]
            )[0][0]

            semantic_chunks.append({
                "text": text,
                "source_file": meta.get("source_file", "unknown"),
                "chunk_id": meta.get("chunk_id", -1),
                "semantic_score": float(sim),
                "bm25_score": 0.0
            })

        return semantic_chunks