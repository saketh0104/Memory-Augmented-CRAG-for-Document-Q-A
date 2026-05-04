from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaStore
from rag.generator import RAGGenerator
from agents.evidence_critic import EvidenceQualityCritic
from agents.query_refiner import QueryRefiner
from agents.llm_intent_router import LLMIntentRouter
from memory.memory_manager import MemoryManager
from retrievers.bm25_retriever import BM25Retriever

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

    def retrieve(self, query: str, use_memory: bool = True):
        # -------- Semantic Retrieval --------
        query_embedding = self.embedder.embed_query(query)

        results = self.vector_db.query_documents(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        semantic_docs = results.get("documents", [[]])[0]
        semantic_meta = results.get("metadatas", [[]])[0]

        semantic_chunks = []
        for text, meta in zip(semantic_docs, semantic_meta):
            semantic_chunks.append({
                "text": text,
                "source_file": meta.get("source_file", "unknown"),
                "chunk_id": meta.get("chunk_id", -1),
                "semantic_score": 1.0  # placeholder (Chroma doesn’t expose raw score easily)
            })

        # -------- BM25 Retrieval --------
        bm25_chunks = self.bm25.search(query, top_k=self.top_k)

        # -------- Fusion --------
        fused = {}

        # Add semantic
        for chunk in semantic_chunks:
            key = (chunk["source_file"], chunk["chunk_id"])
            fused[key] = {
                **chunk,
                "semantic_score": 1.0,
                "bm25_score": 0.0
            }

        # Add BM25
        for chunk in bm25_chunks:
            key = (chunk["source_file"], chunk["chunk_id"])
            if key in fused:
                fused[key]["bm25_score"] = chunk["bm25_score"]
            else:
                fused[key] = {
                    **chunk,
                    "semantic_score": 0.0
                }

        # -------- Normalize Scores --------
        max_bm25 = max([c["bm25_score"] for c in fused.values()] or [1])

        for c in fused.values():
            c["bm25_score"] /= max_bm25

        # -------- Final Scoring --------
        for c in fused.values():
            c["final_score"] = (
                0.7 * c["semantic_score"] +
                0.3 * c["bm25_score"]
            )

        # -------- Sort --------
        ranked = sorted(
            fused.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )

        return ranked[:self.top_k]

    def run(self, query: str):
        print("\n--- QUERY ---")
        print(query)

        intent_config = self.intent_router.classify(query)

        intent = intent_config.get("intent", "FACT_LOOKUP").lower()
        threshold = intent_config.get("threshold", 0.30)
        allow_soft = intent_config.get("allow_soft_aggregation", False)

        self.critic.set_mode(threshold, allow_soft)

        
        episodic_mem, evidence_mem = self.memory.retrieve_memory_context(query)

        
        retrieved_chunks = self.retrieve(query)
        retrieved_chunks.extend(evidence_mem)

        print("\n--- RETRIEVED CHUNKS ---")
        print(len(retrieved_chunks))

        # CRAG
        accepted_chunks, needs_refinement = self.critic.evaluate(
            query, retrieved_chunks
        )

        print("\n--- ACCEPTED CHUNKS ---")
        print(len(accepted_chunks))

        if needs_refinement:
            refined_query = self.refiner.refine(query)
            retrieved_chunks = self.retrieve(refined_query)
            retrieved_chunks.extend(evidence_mem)

            accepted_chunks, _ = self.critic.evaluate(
                refined_query, retrieved_chunks
            )

        # fallback to retrieved if CRAG rejects everything
        final_chunks = accepted_chunks if accepted_chunks else retrieved_chunks

        
        answer = self.generator.generate_answer(
            query=query,
            retrieved_chunks=final_chunks,
            intent=intent,
            episodic_memory=episodic_mem
        )

        if len(accepted_chunks) >= 2:
            self.memory.store_interaction(query, answer, accepted_chunks)

        return {
            "answer": answer,
            "citations": final_chunks,
            "intent": intent
        }
