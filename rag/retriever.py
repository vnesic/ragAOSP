from sentence_transformers import SentenceTransformer
import numpy as np
from .store import VectorStore
from . import config

class Retriever:
    def __init__(self, store: VectorStore):
        self.embed = SentenceTransformer(config.EMBEDDING_MODEL)
        self.store = store

    def embed_text(self, t: str) -> np.ndarray:
        return np.array(self.embed.encode([t], normalize_embeddings=True))[0]

    def query(self, q: str, topk: int):
        qvec = self.embed_text(q)
        hits = self.store.search(qvec, q, topk=topk)
        results = []
        for idx, score in hits:
            m = self.store.meta[idx]
            results.append({"score": score, **m})
        return results
