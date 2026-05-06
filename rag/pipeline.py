from embeddings.embedder import Embedder
from rag.generator import RAGGenerator
from agents.evidence_critic import EvidenceQualityCritic
from agents.query_refiner import QueryRefiner
from agents.llm_intent_router import LLMIntentRouter
from memory.memory_manager import MemoryManager
from retrievers.bm25_retriever import BM25Retriever

from sklearn.metrics.pairwise import cosine_similarity


class RAGPipeline:
    def __init__(self, vector_db, top_k=10):
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
    def retrieve(self, query, bm25_weight=0.3, semantic_weight=0.7, top_k=10):

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_db.query_documents(
            query_embedding=query_embedding,
            top_k=top_k
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        semantic_chunks = []
        for text, meta in zip(docs, metas):
            emb = self.embedder.embed_query(text)

            sim = cosine_similarity([query_embedding], [emb])[0][0]

            semantic_chunks.append({
                "text": text,
                "source_file": meta.get("source_file", "unknown"),
                "chunk_id": meta.get("chunk_id", -1),
                "semantic_score": float(sim),
                "bm25_score": 0.0
            })

        bm25_chunks = self.bm25.search(query, top_k=top_k)

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

        # -------- NORMALIZATION --------
        max_bm25 = max([c["bm25_score"] for c in fused.values()] or [1])
        max_sem = max([c["semantic_score"] for c in fused.values()] or [1])

        for c in fused.values():
            c["bm25_score"] /= max_bm25 if max_bm25 > 0 else 1
            c["semantic_score"] /= max_sem if max_sem > 0 else 1

        for c in fused.values():
            c["final_score"] = (
                semantic_weight * c["semantic_score"] +
                bm25_weight * c["bm25_score"]
            )

        ranked = sorted(fused.values(), key=lambda x: x["final_score"], reverse=True)

        return ranked[:top_k]

    # -------- DEDUP --------
    def _deduplicate(self, chunks):
        seen = set()
        unique = []

        for c in chunks:
            key = (c["source_file"], c["chunk_id"])
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    # ---------------- RUN ----------------
    def run(self, query: str):
        print("\n--- QUERY ---", query)
        intent_cfg = self.intent_router.classify(query)

        intent = intent_cfg["intent"].lower()
        bm25_w = intent_cfg["bm25_weight"]
        sem_w = intent_cfg["semantic_weight"]
        top_k = intent_cfg["top_k"]

        self.critic.set_mode(
            intent_cfg["threshold"],
            intent_cfg["allow_soft_aggregation"]
        )

        episodic_mem, evidence_mem = self.memory.retrieve_memory_context(query)
        
        retrieved = self.retrieve(query, bm25_w, sem_w, top_k)

        # add memory
        for c in evidence_mem:
            c["from_memory"] = True

        retrieved.extend(evidence_mem)

        #FIX: deduplicate
        retrieved = self._deduplicate(retrieved)

        accepted, refine = self.critic.evaluate(query, retrieved)

        if refine:
            rq = self.refiner.refine(query)

            retrieved = self.retrieve(rq, bm25_w, sem_w, top_k)
            retrieved.extend(evidence_mem)
            retrieved = self._deduplicate(retrieved)

            accepted, _ = self.critic.evaluate(rq, retrieved)

        final = accepted if accepted else retrieved

        answer = self.generator.generate_answer(
            query=query,
            retrieved_chunks=final,
            intent=intent,
            episodic_memory=episodic_mem
        )

        if len(accepted) >= 2:
            self.memory.store_interaction(query, answer, accepted)

        return {
            "answer": answer,
            "citations": final,
            "intent": intent
        }