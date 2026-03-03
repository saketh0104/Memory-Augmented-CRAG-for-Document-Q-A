from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaStore
from rag.generator import RAGGenerator
from agents.evidence_critic import EvidenceQualityCritic
from agents.query_refiner import QueryRefiner
from agents.llm_intent_router import LLMIntentRouter
from memory.memory_manager import MemoryManager


class RAGPipeline:
    def __init__(self, top_k: int = 10):
        self.embedder = Embedder()
        self.vector_db = ChromaStore()
        self.generator = RAGGenerator()
        self.memory = MemoryManager(self.vector_db, self.embedder)

        self.critic = EvidenceQualityCritic()
        self.refiner = QueryRefiner()
        self.intent_router = LLMIntentRouter()
        self.top_k = top_k

    def retrieve(self, query: str):

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_db.query_documents(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        retrieved_chunks = []
        for text, meta in zip(documents, metadatas):
            retrieved_chunks.append({
                "text": text,
                "source_file": meta.get("source_file", "unknown"),
                "chunk_id": meta.get("chunk_id", -1)
            })

        # Inject ONLY validated evidence memory
        _, evidence_mem = self.memory.retrieve_memory_context(query)
        retrieved_chunks.extend(evidence_mem)

        return retrieved_chunks

    def run(self, query: str):

        print("\n--- QUERY ---")
        print(query)

       
        intent_config = self.intent_router.classify(query)

        intent = intent_config.get("intent", "FACT_LOOKUP").lower()
        threshold = intent_config.get("threshold", 0.30)
        allow_soft = intent_config.get("allow_soft_aggregation", False)

        self.critic.set_mode(threshold, allow_soft)

        
        retrieved_chunks = self.retrieve(query)

        print("\n--- RETRIEVED CHUNKS ---")
        print(len(retrieved_chunks))

        #CRAG
        accepted_chunks, needs_refinement = self.critic.evaluate(
            query, retrieved_chunks
        )

        print("\n--- ACCEPTED CHUNKS ---")
        print(len(accepted_chunks))

        #Refinement loop
        if needs_refinement:
            refined_query = self.refiner.refine(query)
            retrieved_chunks = self.retrieve(refined_query)
            accepted_chunks, _ = self.critic.evaluate(
                refined_query, retrieved_chunks
            )

        #Fallback
        if not accepted_chunks:
            return {
                "answer": "The document does not contain sufficient information to answer this query.",
                "citations": [],
                "intent": intent
            }


        episodic_mem, _ = self.memory.retrieve_memory_context(query)

        answer = self.generator.generate_answer(
            query=query,
            retrieved_chunks=accepted_chunks,
            intent=intent,
            episodic_memory=episodic_mem
        )

        if (
            "sufficient information" not in answer.lower()
            and len(accepted_chunks) >= 2
        ):
            self.memory.store_interaction(query, answer, accepted_chunks)

        return {
            "answer": answer,
            "citations": accepted_chunks,
            "intent": intent
        }
