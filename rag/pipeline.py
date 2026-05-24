from rag.generator import RAGGenerator
from rag.reranker import CrossEncoderReranker
from agents.evidence_critic import EvidenceQualityCritic
from agents.query_refiner import QueryRefiner
from agents.llm_intent_router import LLMIntentRouter

from memory.memory_manager import MemoryManager
from retrievers.hybrid_retriever import HybridRetriever

from embeddings.embedder import Embedder


class RAGPipeline:
    def __init__(self, vector_db, top_k=10):

        self.embedder = Embedder()
        self.vector_db = vector_db

        self.retriever = HybridRetriever(
            vector_db,
            self.embedder
        )

        self.reranker = CrossEncoderReranker()

        self.generator = RAGGenerator()
        self.memory = MemoryManager(
            self.vector_db,
            self.embedder
        )

        self.critic = EvidenceQualityCritic()
        self.refiner = QueryRefiner()
        self.intent_router = LLMIntentRouter()

        self.top_k = top_k

    def _deduplicate(self, chunks):
        seen = set()
        unique = []

        for c in chunks:
            key = (c["source_file"], c["chunk_id"])

            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    def run(self, query: str):

        print("\n--- QUERY ---", query)

        # -------- INTENT --------
        intent_cfg = self.intent_router.classify(query)

        intent = intent_cfg["intent"].lower()
        bm25_w = intent_cfg["bm25_weight"]
        sem_w = intent_cfg["semantic_weight"]
        top_k = intent_cfg["top_k"]

        # NEW: rerank window
        rerank_k = min(top_k, 8)

        self.critic.set_mode(
            intent_cfg["threshold"],
            intent_cfg["allow_soft_aggregation"]
        )

        # -------- MEMORY --------
        episodic_mem, evidence_mem = (
            self.memory.retrieve_memory_context(query)
        )

        # -------- RETRIEVE --------
        retrieved = self.retriever.search(
            query=query,
            bm25_weight=bm25_w,
            semantic_weight=sem_w,
            top_k=top_k
        )

        # add memory evidence
        for c in evidence_mem:
            c["from_memory"] = True

        retrieved.extend(evidence_mem)
        retrieved = self._deduplicate(retrieved)

        # -------- RERANK --------
        reranked = self.reranker.rerank(
            query=query,
            retrieved_chunks=retrieved,
            top_k=rerank_k
        )

        # -------- CRITIC --------
        accepted, refine = self.critic.evaluate(
            query,
            reranked
        )

        # -------- QUERY REFINEMENT --------
        if refine:

            refined_query = self.refiner.refine(query)

            retrieved = self.retriever.search(
                query=refined_query,
                bm25_weight=bm25_w,
                semantic_weight=sem_w,
                top_k=top_k
            )

            # re-add memory evidence
            for c in evidence_mem:
                c["from_memory"] = True

            retrieved.extend(evidence_mem)
            retrieved = self._deduplicate(retrieved)

            reranked = self.reranker.rerank(
                query=refined_query,
                retrieved_chunks=retrieved,
                top_k=rerank_k
            )

            accepted, _ = self.critic.evaluate(
                refined_query,
                reranked
            )

        # SAFER FALLBACK
        final = accepted if accepted else reranked if reranked else retrieved

        # -------- GENERATION --------
        answer = self.generator.generate_answer(
            query=query,
            retrieved_chunks=final,
            intent=intent,
            episodic_memory=episodic_mem
        )

        # -------- MEMORY STORE --------
        if len(final) >= 2:
            self.memory.store_interaction(
                query,
                answer,
                final
            )

        return {
            "answer": answer,
            "citations": final,
            "intent": intent
        }