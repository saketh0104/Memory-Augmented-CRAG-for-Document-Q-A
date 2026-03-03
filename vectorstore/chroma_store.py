import chromadb

class ChromaStore:
    def __init__(self, persist_dir="vectorstore/chroma"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Main document collection
        self.doc_collection = self.client.get_or_create_collection(
            name="memo_rag_docs"
        )

        # Episodic memory collection
        self.episodic_collection = self.client.get_or_create_collection(
            name="memo_rag_episodic"
        )

        # Evidence memory collection
        self.evidence_collection = self.client.get_or_create_collection(
            name="memo_rag_evidence"
        )

    # -------- Add Documents --------
    def add_documents(self, texts, embeddings, metadatas, ids):
        self.doc_collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    
    # ---- Document Retrieval ----
    def query_documents(self, query_embedding, top_k=5):
        return self.doc_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )


