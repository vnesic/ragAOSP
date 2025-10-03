from pathlib import Path
import re
from typing import Iterable, Tuple

COMMENT_PATTERNS = {
    ".java": re.compile(r"(?s)/\*.*?\*/|//.*?$", re.MULTILINE),
    ".kt":   re.compile(r"(?s)/\*.*?\*/|//.*?$", re.MULTILINE),
    ".xml":  re.compile(r"(?s)<!--.*?-->", re.MULTILINE),
    ".bp":   re.compile(r"#.*?$|//.*?$|/\*.*?\*/", re.MULTILINE|re.DOTALL),
}

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def extract_comment_blocks(text: str, suffix: str) -> Iterable[Tuple[str,int,int]]:
    """
    Yield (snippet, start_line, end_line). For MD/TXT/BP treat full doc;
    for Java/Kotlin/XML, only comments. Keeps line spans for citations.
    """
    if suffix in [".md", ".txt"]:
        lines = text.splitlines()
        if not lines: return
        yield (text, 1, len(lines))
        return

    if suffix == ".bp":
        # Keep full text (bp files carry meaningful context)
        lines = text.splitlines()
        yield (text, 1, len(lines))
        return

    pat = COMMENT_PATTERNS.get(suffix)
    if not pat:
        # For unknown types, skip (or switch to full text if you prefer)
        return

    for m in pat.finditer(text):
        snippet = m.group(0)
        start = text.count("\n", 0, m.start()) + 1
        end   = start + snippet.count("\n")
        cleaned = re.sub(r"(?m)^\s*(//|#)\s?", "", snippet)
        cleaned = re.sub(r"/\*+|\*+/", "", cleaned)
        cleaned = re.sub(r"(?m)^\s*\*\s?", "", cleaned)
        yield (cleaned.strip(), start, end)

def chunk(snippet: str, start_line: int, size: int, overlap: int):
    if not snippet.strip():
        return
    # Rough char-based chunking; you can switch to token-aware chunkers.
    i = 0
    base = start_line
    while i < len(snippet):
        j = min(len(snippet), i + size)
        chunk_text = snippet[i:j]
        # Approx line math
        lines_before = snippet[:i].count("\n")
        lines_in     = chunk_text.count("\n")
        yield chunk_text, base + lines_before, base + lines_before + lines_in
        if j == len(snippet): break
        i = max(i + size - overlap, i + 1)
