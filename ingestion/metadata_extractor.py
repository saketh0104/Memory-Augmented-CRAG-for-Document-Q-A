from datetime import datetime, timezone

datetime.now(timezone.utc).isoformat()

def extract_metadata(filename: str, chunk_id: int):
    return {
        "source_file": filename,
        "chunk_id": chunk_id,
        "ingested_at": datetime.now(timezone.utc).isoformat()
    }