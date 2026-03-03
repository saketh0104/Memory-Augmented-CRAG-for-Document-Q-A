from vectorstore.chroma_store import ChromaStore

db = ChromaStore()
print("Number of stored documents:", db.collection.count())
