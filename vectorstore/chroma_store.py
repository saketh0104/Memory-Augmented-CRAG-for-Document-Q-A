import chromadb
import os


class ChromaStore:
    def __init__(self, persist_dir=None):

        if persist_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            persist_dir = os.path.join(base_dir, "chroma")

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

    # -------- Query --------
    def query_documents(self, query_embedding, top_k=5):
        return self.doc_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
    
    def get_all_documents(self):
        results = self.doc_collection.get(
            include=["documents", "metadatas"]
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        # flatten if nested
        if len(documents) > 0 and isinstance(documents[0], list):
            documents = documents[0]
        if len(metadatas) > 0 and isinstance(metadatas[0], list):
            metadatas = metadatas[0]

        return documents, metadatas