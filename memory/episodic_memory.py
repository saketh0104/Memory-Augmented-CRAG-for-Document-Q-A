import uuid
class EpisodicMemory:
    def __init__(self, chroma_store, embedder):
        self.collection = chroma_store.episodic_collection
        self.embedder = embedder

    def store(self, query, answer):
        self.collection.add(
            documents=[answer],
            embeddings=[self.embedder.embed_query(query)],
            metadatas=[{
                "query": query,
                "answer": answer
            }],
            ids=[f"episodic_{uuid.uuid4()}"]
        )


    def retrieve(self, query, top_k=2):
        if self.collection.count() == 0:
            return []

        query_emb = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        return results.get("metadatas", [[]])[0]

