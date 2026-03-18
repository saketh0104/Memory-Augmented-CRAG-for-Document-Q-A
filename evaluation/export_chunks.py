import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vectorstore.chroma_store import ChromaStore


def export_chunks(vectorstore, metadata_dir="../data/metadata", output_dir="."):
    """
    Export Chroma document chunks into JSON files (50 chunks per file)
    """

    os.makedirs(output_dir, exist_ok=True)

    # -------- Load metadata lookup --------
    metadata_lookup = {}

    metadata_path = Path(metadata_dir)

    if metadata_path.exists():
        for file in metadata_path.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                entries = json.load(f)

                for entry in entries:
                    key = (entry["source_file"], entry["chunk_id"])
                    metadata_lookup[key] = entry

    print(f"Loaded metadata entries: {len(metadata_lookup)}")

    # -------- Fetch Chroma collection --------
    collection = vectorstore.get()

    documents = collection.get("documents", [])
    metadatas = collection.get("metadatas", [])

    print(f"Documents found in vectorstore: {len(documents)}")

    if len(documents) == 0:
        print("No documents found in Chroma collection.")
        return

    exported_chunks = []

    for i, doc in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) else {}

        source_file = meta.get("source_file", "unknown")
        chunk_id = meta.get("chunk_id", i)

        metadata_info = metadata_lookup.get(
            (source_file, chunk_id),
            {}
        )

        exported_chunks.append({
            "global_chunk_index": i,
            "chunk_id": chunk_id,
            "source_file": source_file,
            "ingested_at": metadata_info.get("ingested_at"),
            "content": doc
        })

    # -------- Save batches --------
    batch_size = 50

    for start in range(0, len(exported_chunks), batch_size):
        batch = exported_chunks[start:start + batch_size]

        filename = f"chunks_{start+1}_to_{start+len(batch)}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

        print(f"Saved {filename}")

    print("Export complete.")


if __name__ == "__main__":
    store = ChromaStore()
    vectorstore = store.doc_collection

    export_chunks(vectorstore)