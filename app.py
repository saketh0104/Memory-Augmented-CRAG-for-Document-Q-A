# ---- Suppress warnings early ----
from config import suppress_warnings
suppress_warnings()

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json
from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv

# ---- Project modules ----
from ingestion.loader import load_document
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from ingestion.metadata_extractor import extract_metadata
from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaStore
from rag.pipeline import RAGPipeline


# Load environment variables
load_dotenv()


def create_app():
    app = Flask(__name__)

    # ---------------- CONFIG ----------------
    app.config["UPLOAD_FOLDER"] = "data/raw"
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # ---------------- INIT CORE COMPONENTS ----------------
    embedder = None
    vector_db = ChromaStore()
    rag_pipeline = RAGPipeline(top_k=10)


    SESSION_FOLDER = "data/sessions"
    os.makedirs(SESSION_FOLDER, exist_ok=True)
    app.config["SESSION_FOLDER"] = SESSION_FOLDER


    import uuid
    
    def create_new_session(folder):
        
        session_id = str(uuid.uuid4())
        path = os.path.join(folder, f"{session_id}.json")
        
        session_data = {
            "title": "New Chat",
            "history": []
        }
        
        with open(path, "w") as f:
            json.dump(session_data, f, indent=2)
        
        return session_id

    
    def load_session(folder, session_id):
        path = os.path.join(folder, f"{session_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    @app.route("/load_session/<session_id>", methods=["GET"])
    def load_existing_session(session_id):
        session_data = load_session(app.config["SESSION_FOLDER"], session_id)
        
        if not session_data:
            return jsonify({"error": "Session not found"}), 404
        
        return jsonify(session_data)




    def save_session(folder, session_id, data):
        path = os.path.join(folder, f"{session_id}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


    # ---------------- ROUTES ----------------

    @app.route("/", methods=["GET"])
    def index():
        raw_folder = app.config["UPLOAD_FOLDER"]
        session_folder = app.config["SESSION_FOLDER"]

        documents = []
        if os.path.exists(raw_folder):
            documents = [
                doc for doc in os.listdir(raw_folder)
                if doc.lower().endswith((".pdf", ".docx", ".txt"))
            ]
        
        sessions = []
        
        for f in os.listdir(session_folder):
            if f.endswith(".json"):
                session_id = f.replace(".json", "")
                data = load_session(session_folder, session_id)
                sessions.append({
                    "id": session_id,
                    "title": data.get("title", "Untitled")
                })
        
        return render_template(
            "index.html",
            documents=documents,
            sessions=sessions
        )




    @app.route("/upload", methods=["POST"])
    def upload_file():
    
        if "file" not in request.files:
            return redirect(url_for("index"))

        file = request.files["file"]
        if file.filename == "":
            return redirect(url_for("index"))
        
        # Save uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # -------- INGESTION PIPELINE --------
        raw_text = load_document(file_path)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)
        print("Number of chunks:", len(chunks))
        # Create required directories
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/metadata", exist_ok=True)

        # -------- METADATA (MUST COME FIRST) --------
        metadata = [extract_metadata(file.filename, i) for i in range(len(chunks))]

        # -------- EMBEDDINGS --------
        embeddings = rag_pipeline.embedder.embed_texts(chunks)
        
        # -------- IDS --------
        ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]

        # -------- STORE IN VECTOR DB --------
        vector_db.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )

        print("Documents added successfully.")

        # -------- SAVE METADATA TO DISK --------
        metadata_path = f"data/metadata/{file.filename}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return redirect(url_for("index"))

    @app.route("/query", methods=["POST"])
    def query():
        session_id = request.form.get("session_id")
        user_query = request.form.get("query")
        
        if not session_id:
            return jsonify({"error": "No session id"}), 400
        
        session_data = load_session(app.config["SESSION_FOLDER"], session_id)
        
        if session_data is None:
            return jsonify({"error": "Session not found"}), 404

        result = rag_pipeline.run(user_query)
        
        # If first user message → set session title
        if len(session_data["history"]) == 0:
            session_data["title"] = user_query[:40]
            
        session_data["history"].append({"role": "user", "content": user_query})
        session_data["history"].append({
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"]
        })
        
        save_session(app.config["SESSION_FOLDER"], session_id, session_data)
        
        return jsonify(result)


    @app.route("/new_session", methods=["POST"])
    def new_session():
        session_id = create_new_session(app.config["SESSION_FOLDER"])
        return jsonify({"session_id": session_id})
    

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
        path = os.path.join(app.config["SESSION_FOLDER"], f"{session_id}.json")

        if not os.path.exists(path):
            return jsonify({"error": "Not found"}), 404
        
        with open(path, "r") as f:
            content = json.load(f)
        
        content_meta = {
            "title": new_title,
            "history": content
        }
        
        with open(path, "w") as f:
            json.dump(content_meta, f, indent=2)
        
        return jsonify({"status": "renamed"})


    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False)


