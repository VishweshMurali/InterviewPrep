"""
SkillMapper - RAG Engine (Week 1 Core)
Handles document loading, chunking, embedding, and retrieval
"""

import os

# Disable ChromaDB telemetry before any chromadb import (avoids "capture() takes 1 positional argument" error)
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

# Force CPU for SentenceTransformer to avoid PyTorch "meta tensor" error on some systems
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_DIR   = "./data/chroma_db"
DOCS_DIR     = "./data/documents"
EMBED_MODEL  = "all-MiniLM-L6-v2"
GROQ_MODEL   = "llama-3.1-8b-instant"


class RAGEngine:
    """
    Core RAG Engine:
    - Loads documents from /data/documents
    - Chunks + embeds them into ChromaDB
    - Retrieves relevant chunks for a query
    - Generates answers using Groq LLM

    Heavy imports (langchain, torch, chromadb) are deferred to first use so
    `import rag_engine` is fast and the Streamlit UI renders immediately.
    """

    def __init__(self, embeddings=None):
        self._embeddings = embeddings
        self._llm = None
        self.vectorstore = None
        self.qa_chain    = None

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
    def llm(self):
        if self._llm is None:
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3,
            )
        return self._llm

    # ── Document Ingestion ─────────────────────────────────────────────────────
    def load_documents(self) -> list:
        """Load all PDFs and .txt files from the documents folder."""
        from langchain_community.document_loaders import PyPDFLoader, TextLoader

        docs = []
        docs_path = Path(DOCS_DIR)
        docs_path.mkdir(parents=True, exist_ok=True)

        for file in docs_path.iterdir():
            try:
                if file.suffix == ".pdf":
                    loader = PyPDFLoader(str(file))
                    docs.extend(loader.load())
                    print(f"[OK] Loaded PDF: {file.name}")
                elif file.suffix == ".txt":
                    loader = TextLoader(str(file), encoding="utf-8")
                    docs.extend(loader.load())
                    print(f"[OK] Loaded TXT: {file.name}")
            except Exception as e:
                print(f"[WARNING] Could not load {file.name}: {e}")

        print(f"\n[INFO] Total pages/chunks loaded: {len(docs)}")
        return docs

    def chunk_documents(self, docs: list) -> list:
        """Split documents into overlapping chunks for better retrieval."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(docs)
        print(f"[INFO] Created {len(chunks)} chunks")
        return chunks

    def build_vectorstore(self, chunks: list):
        """Embed chunks and store in ChromaDB."""
        from langchain_community.vectorstores import Chroma
        import chromadb.config

        client_settings = chromadb.config.Settings(anonymized_telemetry=False)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DIR,
            client_settings=client_settings,
        )
        print(f"[INFO] VectorDB built with {self.vectorstore._collection.count()} vectors")

    def load_existing_vectorstore(self) -> bool:
        """Load a previously built ChromaDB if it exists."""
        from langchain_community.vectorstores import Chroma
        import chromadb.config

        if Path(CHROMA_DIR).exists():
            client_settings = chromadb.config.Settings(anonymized_telemetry=False)
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings,
                client_settings=client_settings,
            )
            count = self.vectorstore._collection.count()
            if count > 0:
                print(f"[INFO] Loaded existing VectorDB: {count} vectors")
                return True
        return False

    def setup_qa_chain(self):
        """Wire up the RAG retrieval + generation chain."""
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA

        if not self.vectorstore:
            raise ValueError("VectorStore not initialized. Run build_vectorstore() first.")

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are SkillMapper, an expert AI tutor for Python, DSA, and Machine Learning.

Use ONLY the context below to answer the question.
If the context doesn't contain enough info, say "I need more study material on this topic."

Context:
{context}

Question: {question}

Answer (be clear and educational):""",
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},  # Retrieve top 4 chunks
            ),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True,
        )
        print("[INFO] QA Chain ready!")

    # ── Query ──────────────────────────────────────────────────────────────────
    def query(self, question: str) -> dict:
        """
        Run a RAG query.
        Returns: { answer, sources, question }
        """
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")

        result = self.qa_chain.invoke({"query": question})

        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in result.get("source_documents", [])
        })

        return {
            "question": question,
            "answer":   result["result"],
            "sources":  sources,
        }

    # ── Full Initialization ────────────────────────────────────────────────────
    def initialize(self, force_rebuild: bool = False):
        """
        Smart init: load existing DB or rebuild from documents.
        Set force_rebuild=True to re-ingest all documents.
        """
        if not force_rebuild and self.load_existing_vectorstore():
            self.setup_qa_chain()
            return

        print("[INFO] Building VectorDB from scratch...")
        docs   = self.load_documents()
        if not docs:
            print("[WARNING] No documents found in data/documents/")
            print("    Add .pdf or .txt files there and run again.")
            return
        chunks = self.chunk_documents(docs)
        self.build_vectorstore(chunks)
        self.setup_qa_chain()
        print("\n[OK] RAG Engine fully initialized!\n")
