class EvidenceMemory:
    def __init__(self, chroma_store, embedder):
        self.collection = chroma_store.evidence_collection
        self.embedder = embedder

    def store(self, citations):
        for chunk in citations:
            emb = self.embedder.embed_query(chunk["text"])
            self.collection.add(
                documents=[chunk["text"]],
                embeddings=[emb],
                metadatas=[{
                    "source_file": chunk["source_file"],
                    "chunk_id": chunk["chunk_id"]
                }],
                ids=[f"evidence_{chunk['chunk_id']}"]
            )

    def retrieve(self, query, top_k=3):
        query_emb = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        chunks = []
        for text, meta in zip(documents, metadatas):
            chunks.append({
                "text": text,
                "source_file": meta.get("source_file"),
                "chunk_id": meta.get("chunk_id")
            })

        return chunks
