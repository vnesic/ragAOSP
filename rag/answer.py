from textwrap import dedent

SYSTEM = "You answer Android/AOSP questions using the provided context. If unsure, say you don't know. Always cite file paths and line ranges."

def build_prompt(question: str, contexts: list[str]):
    joined = "\n\n".join(f"[{i+1}]\n{c}" for i,c in enumerate(contexts))
    return dedent(f"""
    {SYSTEM}

    Context:
    {joined}

    Question: {question}

    Answer concisely. When you make a factual claim, include inline citations like [path:lineStart-lineEnd].
    """).strip()
