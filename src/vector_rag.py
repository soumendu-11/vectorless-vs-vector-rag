"""
vector_rag.py
-------------
Section-Based Chunking + FAISS In-Memory Vector RAG.

Pipeline:
  1. Load the PageIndex structure JSON → section boundaries
  2. Extract text per section using Docling (reuses qa_system cache)
  3. Embed each chunk with sentence-transformers (all-MiniLM-L6-v2, local, no API key)
  4. Build a FAISS flat-L2 index entirely in memory
  5. Answer the question: embed query → top-k chunks → Azure OpenAI

Chunking strategy: one chunk per unique page-range in the PageIndex tree.
This respects semantic section boundaries rather than arbitrary token windows.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Project root
ROOT_DIR = Path(__file__).parent.parent
_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

load_dotenv(ROOT_DIR / ".env")

from openai import AzureOpenAI  # noqa: E402

_llm_client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("API_VERSION"),
)
DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

# Sentence-transformer model (downloaded once, then cached by HuggingFace)
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"[vector_rag] Loading embedding model: {_EMBED_MODEL_NAME}…")
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        print("[vector_rag] Embedding model loaded.")
    return _embed_model


# ── Step 1: Section-based chunking ──────────────────────────────────────────

def create_section_chunks(pdf_path: Path, structure: dict) -> list[dict]:
    """
    Create one text chunk per unique page-range found in the PageIndex structure.

    Each chunk:
        {node_id, title, pages, start_page, end_page, text}

    Text is extracted with Docling (markdown tables preserved), reusing
    the shared cache from qa_system so the PDF is only parsed once.
    """
    from qa_system import flatten_nodes, extract_pages

    flat = flatten_nodes(structure.get("structure", []))

    # Deduplicate by page range — keep first node that maps to each range
    seen: dict[tuple, dict] = {}
    for node in flat:
        if not node.get("node_id"):
            continue
        key = (node["start_index"], node["end_index"])
        if key not in seen:
            seen[key] = node

    chunks = []
    for (s, e), node in sorted(seen.items()):
        text = extract_pages(pdf_path, [(s, e)])
        if not text.strip():
            continue
        chunks.append({
            "node_id": node["node_id"],
            "title": node["title"],
            "pages": f"{s}–{e}",
            "start_page": s,
            "end_page": e,
            "text": text,
        })

    print(f"[vector_rag] Created {len(chunks)} section chunks from {len(seen)} unique page ranges.")
    return chunks


# ── Step 2 & 3: Embed + FAISS ───────────────────────────────────────────────

def build_vector_store(chunks: list[dict]) -> tuple[faiss.Index, np.ndarray]:
    """
    Embed all chunks and build an in-memory FAISS flat-L2 index.

    Returns:
        (faiss_index, embeddings_array)
    """
    model = _get_embed_model()
    texts = [f"{c['title']}\n\n{c['text']}" for c in chunks]

    print(f"[vector_rag] Embedding {len(texts)} chunks…")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"[vector_rag] FAISS index built — {index.ntotal} vectors, dim={dim}.")
    return index, embeddings


def retrieve_chunks(
    question: str,
    chunks: list[dict],
    index: faiss.Index,
    top_k: int = 5,
) -> list[dict]:
    """
    Embed the question and return the top-k most similar chunks.
    """
    model = _get_embed_model()
    q_vec = model.encode([question], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_vec, top_k)
    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
    print(f"[vector_rag] Retrieved {len(retrieved)} chunks for the query.")
    return retrieved


# ── Step 4: Generate answer ──────────────────────────────────────────────────

def _build_context(retrieved: list[dict]) -> str:
    parts = []
    for c in retrieved:
        parts.append(f"--- [{c['node_id']}] {c['title']} (pages {c['pages']}) ---\n{c['text']}")
    return "\n\n".join(parts)


def generate_vector_answer(question: str, retrieved: list[dict]) -> str:
    """
    Generate a bullet-point answer using only the retrieved chunks.
    Uses the same prompt format as qa_system for a fair comparison.
    """
    context = _build_context(retrieved)

    system_prompt = (
        "You are a financial analyst assistant. "
        "Answer the user's question using ONLY the provided document excerpts.\n\n"
        "FORMAT RULES — follow strictly:\n"
        "1. Start with a one-sentence headline summary.\n"
        "2. Use top-level bullet points (•) for each major theme or metric.\n"
        "3. Use indented sub-bullets (  ◦) for segment-level or supporting details.\n"
        "4. Always include exact numbers, percentages, and dollar amounts.\n"
        "5. Do NOT write paragraphs — every piece of information must be a bullet.\n"
        "6. End with a '📌 Key Takeaway' bullet summarising the single most important finding."
    )

    response = _llm_client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nDocument Excerpts:\n{context}"},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def vector_rag_answer(
    question: str,
    pdf_path: Path,
    structure: dict,
    top_k: int = 5,
) -> dict:
    """
    Full Section-Based Vector RAG pipeline.

    Returns:
        {
            "question": str,
            "retrieved_chunks": list,    # metadata only
            "context": str,              # full retrieved text (for evaluation)
            "answer": str,
        }
    """
    # 1. Chunk
    chunks = create_section_chunks(pdf_path, structure)

    # 2 & 3. Embed + index
    index, _ = build_vector_store(chunks)

    # 4. Retrieve
    retrieved = retrieve_chunks(question, chunks, index, top_k=top_k)

    # 5. Answer
    print("[vector_rag] Generating answer…")
    answer = generate_vector_answer(question, retrieved)

    return {
        "question": question,
        "retrieved_chunks": [
            {"node_id": c["node_id"], "title": c["title"], "pages": c["pages"]}
            for c in retrieved
        ],
        "context": _build_context(retrieved),
        "answer": answer,
    }


if __name__ == "__main__":
    from index_pdf import find_pdf, DATA_DIR, RESULTS_DIR
    from qa_system import load_structure, find_structure_json
    from qa_system import DEFAULT_QUESTION as QUESTION

    pdf_path = find_pdf(DATA_DIR)
    structure = load_structure(find_structure_json(RESULTS_DIR))

    result = vector_rag_answer(QUESTION, pdf_path, structure)

    print("\n" + "=" * 60)
    print("VECTOR RAG ANSWER")
    print("=" * 60)
    print(f"Question: {result['question']}\n")
    print("Retrieved chunks:")
    for c in result["retrieved_chunks"]:
        print(f"  [{c['node_id']}] {c['title']}  (pages {c['pages']})")
    print(f"\nAnswer:\n{result['answer']}")
    print("=" * 60)
