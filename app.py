import os
import json
from os import path
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv

# ---- Project modules ----
from ingestion.loader import load_document
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from ingestion.metadata_extractor import extract_metadata
from vectorstore.chroma_store import ChromaStore
from rag.pipeline import RAGPipeline

# Load env
load_dotenv()


def create_app():
    app = Flask(__name__)

    # ---------------- CONFIG ----------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "raw")
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    SESSION_FOLDER = os.path.join(BASE_DIR, "data", "sessions")
    app.config["SESSION_FOLDER"] = SESSION_FOLDER
    os.makedirs(SESSION_FOLDER, exist_ok=True)

    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["SESSION_FOLDER"], exist_ok=True)

    # ---------------- INIT CORE ----------------
    vector_db = ChromaStore()
    rag_pipeline = RAGPipeline(vector_db=vector_db, top_k=10)

    # -------- BUILD BM25 ON STARTUP --------
    documents, metadatas = vector_db.get_all_documents()
    if documents:
        rag_pipeline.bm25.build_index(documents, metadatas)
        print(f"[BM25] Loaded {len(documents)} documents on startup.")
    else:
        print("[BM25] No documents found on startup.")

    # ---------------- SESSION UTILS ----------------

    def create_new_session():
        session_id = str(uuid.uuid4())
        path = os.path.join(app.config["SESSION_FOLDER"], f"{session_id}.json")

        data = {
            "id": session_id,
            "title": "New Chat",
            "pinned": False,
            "history": []
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return session_id

    def load_session(session_id):
        path = os.path.join(app.config["SESSION_FOLDER"], f"{session_id}.json")
        print("SESSION PATH:", path)
        print("EXISTS:", os.path.exists(path))
        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            return json.load(f)
        
        

    def save_session(session_id, data):
        path = os.path.join(app.config["SESSION_FOLDER"], f"{session_id}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ---------------- ROUTES ----------------

    @app.route("/", methods=["GET"])
    def index():
        docs = []
        sessions = []

        if os.path.exists(app.config["UPLOAD_FOLDER"]):
            docs = [
                f for f in os.listdir(app.config["UPLOAD_FOLDER"])
                if f.lower().endswith((".pdf", ".docx", ".txt"))
            ]

        for f in sorted(os.listdir(app.config["SESSION_FOLDER"]), reverse=True):
            if f.endswith(".json"):
                sid = f.replace(".json", "")
                data = load_session(sid)
                if data:
                    sessions.append({
                        "id": sid,
                        "title": data.get("title", "Untitled")
                    })

        return render_template(
            "index.html",
            documents=docs,
            sessions=sessions
        )

    # ---------------- FILE UPLOAD ----------------

    @app.route("/upload", methods=["POST"])
    def upload_file():

        if "file" not in request.files:
            return redirect(url_for("index"))

        file = request.files["file"]

        if file.filename == "":
            return redirect(url_for("index"))

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # -------- INGESTION --------
        raw_text = load_document(file_path)
        cleaned_text = clean_text(raw_text)

        chunks = chunk_text(cleaned_text, rag_pipeline.embedder)
        print("Chunks created:", len(chunks))

        metadata = [
            extract_metadata(file.filename, i)
            for i in range(len(chunks))
        ]

        embeddings = rag_pipeline.embedder.embed_texts(chunks)

        ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]

        vector_db.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )

        #UPDATE BM25 INDEX
        documents, metadatas = vector_db.get_all_documents()
        rag_pipeline.bm25.build_index(documents, metadatas)

        print("[UPLOAD] Document indexed successfully.")

        # Save metadata snapshot
        os.makedirs("data/metadata", exist_ok=True)
        with open(f"data/metadata/{file.filename}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return jsonify({
    "status": "uploaded",
    "filename": file.filename,
    "chunks": len(chunks)
    })

    # ---------------- QUERY ----------------

    @app.route("/query", methods=["POST"])
    def query():
        session_id = request.form.get("session_id")
        user_query = request.form.get("query")

        if not session_id:
            return jsonify({"error": "No session id"}), 400

        session_data = load_session(session_id)
        if session_data is None:
            return jsonify({"error": "Session not found"}), 404


        session_history = session_data.get("history", [])

        result = rag_pipeline.run(user_query)

        # Ensure history exists
        if not isinstance(session_data.get("history"), list):
            session_data["history"] = []

        # Set title on first query
        if len(session_data["history"]) == 0:
            session_data["title"] = user_query[:40]

        session_data["history"].append({
            "role": "user",
            "content": user_query
        })

        session_data["history"].append({
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"]
        })

        save_session(session_id, session_data)

        return jsonify(result)

    # ---------------- SESSION ROUTES ----------------

    @app.route("/load_session/<session_id>", methods=["GET"])
    def load_existing_session(session_id):
        session_data = load_session(session_id)
        if session_data is None:
            return jsonify({"error": "Session not found"}), 404

        return jsonify(session_data)

    @app.route("/new_session", methods=["POST"])
    def new_session():
        sid = create_new_session()
        return jsonify({"session_id": sid})

    @app.route("/delete_session/<session_id>", methods=["POST"])
    def delete_session(session_id):
        path = os.path.join(app.config["SESSION_FOLDER"], f"{session_id}.json")

        if os.path.exists(path):
            os.remove(path)
            return jsonify({"status": "deleted"})

        return jsonify({"error": "Session not found"}), 404

    @app.route("/rename_session/<session_id>", methods=["POST"])
    def rename_session(session_id):
        data = request.get_json()
        new_title = data.get("title")

        if not new_title or not new_title.strip():
            return jsonify({"error": "Invalid title"}), 400

        session_data = load_session(session_id)
        if session_data is None:
            return jsonify({"error": "Not found"}), 404

        session_data["title"] = new_title
        save_session(session_id, session_data)

        return jsonify({"status": "renamed"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=False)