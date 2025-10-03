# rag/indexing.py
from pathlib import Path
from typing import List
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from fnmatch import fnmatch

from . import config
from .parsing import read_text, extract_comment_blocks, chunk
from .store import VectorStore


def _norm_abs(p: Path) -> str:
    return p.resolve().as_posix()


def _norm_rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return p.name  # fallback


def _match_any(path_str: str, patterns: List[str]) -> bool:
    return any(fnmatch(path_str, pat) for pat in patterns)


def _should_include(p: Path, root: Path) -> bool:
    if not config.INCLUDE_GLOBS:
        return True
    abs_s = _norm_abs(p)
    rel_s = _norm_rel(p, root)
    return _match_any(abs_s, config.INCLUDE_GLOBS) or _match_any(rel_s, config.INCLUDE_GLOBS)


def _should_exclude(p: Path, root: Path) -> bool:
    if not config.EXCLUDE_GLOBS:
        return False
    abs_s = _norm_abs(p)
    rel_s = _norm_rel(p, root)
    return _match_any(abs_s, config.EXCLUDE_GLOBS) or _match_any(rel_s, config.EXCLUDE_GLOBS)


def collect_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if not _should_include(p, root):
            continue
        if _should_exclude(p, root):
            continue
        files.append(p)
    return files


def build_index():
    root = Path(config.AOSP_ROOT or "")
    if not root.exists():
        raise RuntimeError(f"AOSP_ROOT does not exist: {root}")

    files = collect_files(root)
    chunks, metas = [], []

    for p in tqdm(files, desc="Scanning"):
        suffix = p.suffix.lower()
        txt = read_text(p)
        if not txt:
            continue
        for snippet, start, end in extract_comment_blocks(txt, suffix):
            for piece, sline, eline in chunk(snippet, start, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                chunks.append(piece)
                metas.append({
                    "path": _norm_rel(p, root),
                    "start_line": sline,
                    "end_line": eline,
                    "text": piece
                })

    if not chunks:
        raise RuntimeError("No chunks found. Adjust INCLUDE_GLOBS/EXCLUDE_GLOBS in .env.")

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    vecs = embedder.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype="float32")

    store = VectorStore(config.INDEX_DIR)
    store.build(vecs, metas, [m["text"] for m in metas])
    store.save()
