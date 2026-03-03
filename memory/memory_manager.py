from .episodic_memory import EpisodicMemory
from .evidence_memory import EvidenceMemory


class MemoryManager:
    def __init__(self, chroma_store, embedder):
        self.episodic = EpisodicMemory(chroma_store, embedder)
        self.evidence = EvidenceMemory(chroma_store, embedder)

    def retrieve_memory_context(self, query):
        episodic_context = self.episodic.retrieve(query)
        evidence_context = self.evidence.retrieve(query)

        return episodic_context, evidence_context

    def store_interaction(self, query, answer, citations):
        self.episodic.store(query, answer)
        self.evidence.store(citations)
