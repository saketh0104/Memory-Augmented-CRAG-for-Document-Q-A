from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaStore

embedder = Embedder()
db = ChromaStore()

query = "What is the purpose of this document?"
q_emb = embedder.embed_query(query)

results = db.query(q_emb, top_k=3)
print(results["documents"])
