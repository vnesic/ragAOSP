import os
from dotenv import load_dotenv

load_dotenv()

AOSP_ROOT       = os.getenv("AOSP_ROOT")
INDEX_DIR       = os.getenv("INDEX_DIR", ".rag_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL    = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
INCLUDE_GLOBS   = [g.strip() for g in os.getenv("INCLUDE_GLOBS", "").split(",") if g.strip()]
EXCLUDE_GLOBS   = [g.strip() for g in os.getenv("EXCLUDE_GLOBS", "").split(",") if g.strip()]
TOPK            = int(os.getenv("TOPK", "12"))
RERANK_TOPK     = int(os.getenv("RERANK_TOPK", "6"))
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "200"))
