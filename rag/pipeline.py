from embeddings.embedder import Embedder
from rag.generator import RAGGenerator
from agents.evidence_critic import EvidenceQualityCritic
from agents.query_refiner import QueryRefiner
from agents.llm_intent_router import LLMIntentRouter
from memory.memory_manager import MemoryManager
from retrievers.bm25_retriever import BM25Retriever

from sklearn.metrics.pairwise import cosine_similarity


class RAGPipeline:
    def __init__(self, vector_db, top_k: int = 10):
        self.embedder = Embedder()
        self.vector_db = vector_db
        self.generator = RAGGenerator()
        self.memory = MemoryManager(self.vector_db, self.embedder)
        self.bm25 = BM25Retriever()

        self.critic = EvidenceQualityCritic()
        self.refiner = QueryRefiner()
        self.intent_router = LLMIntentRouter()

        self.top_k = top_k

    # ---------------- RETRIEVAL ----------------
    def retrieve(
        self,
        query: str,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        top_k: int = 10
    ):
        # -------- Embed query --------
        query_embedding = self.embedder.embed_query(query)

        # -------- Semantic Retrieval --------
        results = self.vector_db.query_documents(
            query_embedding=query_embedding,
            top_k=top_k
        )

        semantic_docs = results.get("documents", [[]])[0]
        semantic_meta = results.get("metadatas", [[]])[0]

        semantic_chunks = []

        for text, meta in zip(semantic_docs, semantic_meta):
            chunk_embedding = self.embedder.embed_query(text)

            sim = cosine_similarity(
                [query_embedding],
                [chunk_embedding]
            )[0][0]

            semantic_chunks.append({
                "text": text,
                "source_file": meta.get("source_file", "unknown"),
                "chunk_id": meta.get("chunk_id", -1),
                "semantic_score": float(sim),
                "bm25_score": 0.0
            })

        # -------- BM25 Retrieval --------
        bm25_chunks = self.bm25.search(query, top_k=top_k)

        # -------- Fusion --------
        fused = {}

        # Add semantic
        for chunk in semantic_chunks:
            key = (chunk["source_file"], chunk["chunk_id"])
            fused[key] = chunk

        # Add BM25
        for chunk in bm25_chunks:
            key = (chunk["source_file"], chunk["chunk_id"])

            if key in fused:
                fused[key]["bm25_score"] = chunk["bm25_score"]
            else:
                fused[key] = {
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "chunk_id": chunk["chunk_id"],
                    "semantic_score": 0.0,
                    "bm25_score": chunk["bm25_score"]
                }

        # -------- Normalize Scores --------
        max_bm25 = max([c["bm25_score"] for c in fused.values()] or [1])
        max_semantic = max([c["semantic_score"] for c in fused.values()] or [1])

        for c in fused.values():
            c["bm25_score"] = c["bm25_score"] / max_bm25 if max_bm25 > 0 else 0
            c["semantic_score"] = c["semantic_score"] / max_semantic if max_semantic > 0 else 0

        # -------- Final Score --------
        for c in fused.values():
            c["final_score"] = (
                semantic_weight * c["semantic_score"] +
                bm25_weight * c["bm25_score"]
            )

        # -------- Sort --------
        ranked = sorted(
            fused.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )

        return ranked[:top_k]

    # ---------------- MAIN PIPELINE ----------------
    def run(self, query: str):
        print("\n--- QUERY ---")
        print(query)

        # -------- Intent --------
        intent_config = self.intent_router.classify(query)

        intent = intent_config.get("intent", "FACT_LOOKUP").lower()
        threshold = intent_config.get("threshold", 0.30)
        allow_soft = intent_config.get("allow_soft_aggregation", False)
        bm25_weight = intent_config.get("bm25_weight", 0.3)
        semantic_weight = intent_config.get("semantic_weight", 0.7)
        dynamic_top_k = intent_config.get("top_k", self.top_k)

        print("\n--- RETRIEVAL CONFIG ---")
        print(f"Intent: {intent}")
        print(f"BM25: {bm25_weight}, Semantic: {semantic_weight}")
        print(f"Top-K: {dynamic_top_k}")

        self.critic.set_mode(threshold, allow_soft)

        # -------- Memory --------
        episodic_mem, evidence_mem = self.memory.retrieve_memory_context(query)

        # -------- Retrieval --------
        retrieved_chunks = self.retrieve(
            query,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            top_k=dynamic_top_k
        )

        # Tag memory chunks
        for c in evidence_mem:
            c["from_memory"] = True

        retrieved_chunks.extend(evidence_mem)

        print("\n--- RETRIEVED CHUNKS ---")
        print(len(retrieved_chunks))

        # -------- CRAG --------
        accepted_chunks, needs_refinement = self.critic.evaluate(
            query, retrieved_chunks
        )

        print("\n--- ACCEPTED CHUNKS ---")
        print(len(accepted_chunks))

        # -------- Refinement --------
        if needs_refinement:
            print("\n--- QUERY REFINEMENT TRIGGERED ---")

            refined_query = self.refiner.refine(query)

            retrieved_chunks = self.retrieve(
                refined_query,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
                top_k=dynamic_top_k
            )

            retrieved_chunks.extend(evidence_mem)

            accepted_chunks, _ = self.critic.evaluate(
                refined_query, retrieved_chunks
            )

        # -------- Final Chunks --------
        final_chunks = accepted_chunks if accepted_chunks else retrieved_chunks

        # -------- Generation --------
        answer = self.generator.generate_answer(
            query=query,
            retrieved_chunks=final_chunks,
            intent=intent,
            episodic_memory=episodic_mem
        )

        # -------- Memory Update --------
        if len(accepted_chunks) >= 2:
            self.memory.store_interaction(query, answer, accepted_chunks)

        return {
            "answer": answer,
            "citations": final_chunks,
            "intent": intent
        }