"""
SkillMapper - Document Manager
================================
Handles MULTIPLE file uploads:
- Accepts PDFs, TXT, MD files
- Extracts and chunks all content
- Stores everything in ChromaDB with source metadata
- Tracks which files have been indexed
- Supports incremental updates (add new files without re-indexing old ones)
"""

import os

# Disable ChromaDB telemetry before any chromadb import (avoids "capture() takes 1 positional argument" error)
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Force CPU for SentenceTransformer to avoid PyTorch "meta tensor" error on some systems
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

CHROMA_DIR   = "./data/chroma_db"
DOCS_DIR     = "./data/documents"
INDEX_FILE   = "./logs/indexed_files.json"
EMBED_MODEL  = "all-MiniLM-L6-v2"


def _file_hash(path: str) -> str:
    """MD5 hash of file contents — used to detect duplicates."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


class DocumentManager:
    """
    Manages all uploaded documents with incremental indexing.

    Key features:
    - Never re-indexes the same file twice (hash-based dedup)
    - Tracks per-file metadata (name, upload time, chunk count)
    - Supports removing files from the index
    - Returns per-source stats for the UI
    """

    def __init__(self, embeddings=None):
        Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)

        self._embeddings = embeddings
        self._vectorstore = None
        self.index_data = self._load_index()

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            self._embeddings = SentenceTransformerEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": "cpu"},
            )
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            from langchain_community.vectorstores import Chroma
            import chromadb.config
            client_settings = chromadb.config.Settings(anonymized_telemetry=False)
            self._vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings,
                client_settings=client_settings,
            )
        return self._vectorstore

    @vectorstore.setter
    def vectorstore(self, value):
        self._vectorstore = value

    # ── Index Management ───────────────────────────────────────────────────────
    def _load_index(self) -> dict:
        if Path(INDEX_FILE).exists():
            try:
                with open(INDEX_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_index(self):
        Path("./logs").mkdir(parents=True, exist_ok=True)
        with open(INDEX_FILE, "w") as f:
            json.dump(self.index_data, f, indent=2)

    # ── File Upload ────────────────────────────────────────────────────────────
    def add_file_from_bytes(self, file_bytes: bytes, filename: str) -> dict:
        """
        Process an uploaded file (from Streamlit uploader).
        Returns status dict: {success, filename, chunks_added, already_existed}
        """
        import tempfile

        suffix = Path(filename).suffix.lower()
        if suffix not in (".pdf", ".txt", ".md"):
            return {"success": False, "error": f"Unsupported type: {suffix}",
                    "filename": filename, "chunks_added": 0}

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            file_hash = hashlib.md5(file_bytes).hexdigest()

            # Check duplicate
            if file_hash in self.index_data:
                return {
                    "success": True, "filename": filename,
                    "chunks_added": 0, "already_existed": True,
                    "message": f"'{filename}' already indexed — skipping"
                }

            # Load and chunk
            docs   = self._load_single_file(tmp_path, filename)
            chunks = self._chunk_documents(docs)

            if not chunks:
                return {"success": False, "filename": filename,
                        "chunks_added": 0, "error": "No text could be extracted"}

            # Add to vectorstore
            self.vectorstore.add_documents(chunks)

            # Update index
            self.index_data[file_hash] = {
                "filename":    filename,
                "chunks":      len(chunks),
                "uploaded_at": datetime.now().isoformat(),
                "size_bytes":  len(file_bytes),
            }
            self._save_index()

            # Also save permanent copy
            save_path = Path(DOCS_DIR) / filename
            save_path.write_bytes(file_bytes)

            return {
                "success": True, "filename": filename,
                "chunks_added": len(chunks), "already_existed": False,
                "message": f"[OK] '{filename}' indexed - {len(chunks)} chunks added"
            }

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def add_file_from_path(self, file_path: str) -> dict:
        """Process a file that already exists on disk."""
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}",
                    "filename": path.name, "chunks_added": 0}

        file_bytes = path.read_bytes()
        return self.add_file_from_bytes(file_bytes, path.name)

    def bulk_index_folder(self) -> list:
        """Index all files in data/documents/ that haven't been indexed yet."""
        # If index says we have files but Chroma has no vectors, we're out of sync (e.g. Chroma was cleared) — force re-index
        if self.vectorstore._collection.count() == 0 and len(self.index_data) > 0:
            self.index_data = {}
            self._save_index()
        results = []
        docs_path = Path(DOCS_DIR)
        for f in docs_path.iterdir():
            if f.suffix.lower() in (".pdf", ".txt", ".md"):
                result = self.add_file_from_path(str(f))
                results.append(result)
        return results

    # ── Document Loading ───────────────────────────────────────────────────────
    def _load_single_file(self, file_path: str, display_name: str) -> list:
        """Load a single file and return LangChain documents."""
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain.schema import Document

        path   = Path(file_path)
        suffix = path.suffix.lower()
        docs   = []

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
                docs   = loader.load()
            elif suffix in (".txt", ".md"):
                loader = TextLoader(str(path), encoding="utf-8")
                docs   = loader.load()

            # Override source metadata with display name
            for doc in docs:
                doc.metadata["source"]       = display_name
                doc.metadata["display_name"] = display_name

        except Exception as e:
            print(f"[WARNING] Could not load {display_name}: {e}")

        return docs

    def _chunk_documents(self, docs: list) -> list:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "],
        )
        return splitter.split_documents(docs)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    def get_retriever(self, k: int = 4):
        """Return a retriever for use in the RAG chain."""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def query_direct(self, query: str, k: int = 4) -> list:
        """Direct similarity search — returns list of (doc, score)."""
        return self.vectorstore.similarity_search_with_score(query, k=k)

    # ── Stats & Metadata ───────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Return stats about indexed documents. Avoids loading vectorstore if it hasn't been used yet."""
        total_chunks = sum(v["chunks"] for v in self.index_data.values())
        if self._vectorstore is not None:
            vec_count = self._vectorstore._collection.count()
        else:
            vec_count = 0
        return {
            "total_files":   len(self.index_data),
            "total_chunks":  total_chunks,
            "total_vectors": vec_count,
            "files": [
                {
                    "filename":    v["filename"],
                    "chunks":      v["chunks"],
                    "uploaded_at": v["uploaded_at"],
                    "size_kb":     round(v.get("size_bytes", 0) / 1024, 1),
                }
                for v in self.index_data.values()
            ]
        }

    def get_indexed_filenames(self) -> list[str]:
        return [v["filename"] for v in self.index_data.values()]

    def is_empty(self) -> bool:
        return self.vectorstore._collection.count() == 0

    def remove_file(self, filename: str) -> bool:
        """Remove a file from the index (marks it for re-indexing next time)."""
        to_remove = [k for k, v in self.index_data.items() if v["filename"] == filename]
        for k in to_remove:
            del self.index_data[k]
        self._save_index()
        return len(to_remove) > 0

    def reset_all(self):
        """Clear all indexed documents."""
        import shutil
        import time
        
        # Close the vectorstore to release file handles
        try:
            if hasattr(self.vectorstore, '_client'):
                del self.vectorstore._client
            del self.vectorstore
        except:
            pass
        
        # Try to remove the directory with retries (Windows file locking issue)
        if Path(CHROMA_DIR).exists():
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(CHROMA_DIR)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)  # Wait for file handles to release
                    else:
                        print(f"[WARNING] Could not delete ChromaDB. Please close the app and manually delete: {CHROMA_DIR}")
                        # Continue anyway to clear the index
        
        self.index_data = {}
        self._save_index()
        
        # Recreate vectorstore
        from langchain_community.vectorstores import Chroma
        import chromadb.config
        client_settings = chromadb.config.Settings(anonymized_telemetry=False)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            client_settings=client_settings,
        )
