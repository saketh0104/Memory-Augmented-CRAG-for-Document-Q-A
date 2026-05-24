from retrievers.semantic_retriever import SemanticRetriever
from retrievers.bm25_retriever import BM25Retriever


class HybridRetriever:
    def __init__(self, vector_db, embedder):
        self.semantic = SemanticRetriever(vector_db, embedder)
        self.bm25 = BM25Retriever()

    def _deduplicate(self, chunks):
        seen = set()
        unique = []

        for c in chunks:
            key = (c["source_file"], c["chunk_id"])

            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    def search(
        self,
        query,
        bm25_weight=0.3,
        semantic_weight=0.7,
        top_k=10
    ):

        semantic_chunks = self.semantic.search(
            query=query,
            top_k=top_k
        )

        bm25_chunks = self.bm25.search(
            query=query,
            top_k=top_k
        )

        # -------- FUSION --------
        fused = {}

        for c in semantic_chunks:
            key = (c["source_file"], c["chunk_id"])
            fused[key] = c

        for c in bm25_chunks:
            key = (c["source_file"], c["chunk_id"])

            if key in fused:
                fused[key]["bm25_score"] = c["bm25_score"]

            else:
                fused[key] = {
                    "text": c["text"],
                    "source_file": c["source_file"],
                    "chunk_id": c["chunk_id"],
                    "semantic_score": 0.0,
                    "bm25_score": c["bm25_score"]
                }

        # -------- NORMALIZE --------
        max_bm25 = max(
            [c["bm25_score"] for c in fused.values()] or [1]
        )

        max_semantic = max(
            [c["semantic_score"] for c in fused.values()] or [1]
        )

        for c in fused.values():
            c["bm25_score"] /= max_bm25 if max_bm25 > 0 else 1
            c["semantic_score"] /= max_semantic if max_semantic > 0 else 1

        # -------- FINAL SCORE --------
        for c in fused.values():
            c["final_score"] = (
                semantic_weight * c["semantic_score"] +
                bm25_weight * c["bm25_score"]
            )

        ranked = sorted(
            fused.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )

        ranked = self._deduplicate(ranked)

        return ranked[:top_k]