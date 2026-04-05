"""
qa_system.py
------------
Vectorless RAG Q&A system built on top of the PageIndex JSON structure.

Workflow:
  1. Load the hierarchical JSON index produced by index_pdf.py
  2. Flatten the node tree into a compact text summary
  3. Ask Azure OpenAI to identify the most relevant node_ids for the question
  4. Extract the corresponding page ranges using Docling (richer text + table extraction)
  5. Ask Azure OpenAI to answer the question in structured bullet-point format

All API calls go through the openai SDK pointed at Azure OpenAI.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")  # suppress numpy compat warnings from docling deps
from pathlib import Path
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter

# Project root is two levels up from src/
ROOT_DIR = Path(__file__).parent.parent

# Ensure src/ is on the path so the vendored `pageindex` package is importable
_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Module-level cache: avoid re-parsing the PDF on every call
_docling_cache: dict = {}

# Load environment variables from .env at project root
load_dotenv(ROOT_DIR / ".env")

from openai import AzureOpenAI  # noqa: E402 (after dotenv)

# ── Azure OpenAI client ──────────────────────────────────────────────────────
_client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("API_VERSION"),
)
DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_QUESTION = (
    "How did the company's total revenue and operating margin change in Q1 FY25 "
    "compared to the previous year, and which business segments contributed most "
    "to this change?"
)


# ── Structure helpers ────────────────────────────────────────────────────────

def load_structure(json_path: Path) -> dict:
    """Load the PageIndex JSON from disk."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_nodes(nodes: list, flat: list | None = None, depth: int = 0) -> list:
    """
    Recursively flatten the nested node tree into a list of dicts:
      {node_id, title, summary, start_index, end_index, depth}
    """
    if flat is None:
        flat = []
    for node in nodes:
        flat.append({
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "summary": node.get("summary", ""),
            "start_index": node.get("start_index"),
            "end_index": node.get("end_index"),
            "depth": depth,
        })
        children = node.get("nodes") or node.get("children") or []
        flatten_nodes(children, flat, depth + 1)
    return flat


def build_structure_text(flat_nodes: list, max_summary_chars: int = 300) -> str:
    """Build a compact text representation of all nodes for the LLM to reason over."""
    lines = []
    for n in flat_nodes:
        indent = "  " * n["depth"]
        summary_snippet = (n["summary"] or "")[:max_summary_chars].replace("\n", " ")
        lines.append(
            f"{indent}[{n['node_id']}] {n['title']} "
            f"(pages {n['start_index']}–{n['end_index']})\n"
            f"{indent}  Summary: {summary_snippet}"
        )
    return "\n".join(lines)


# ── Step B: LLM navigation ───────────────────────────────────────────────────

def identify_relevant_nodes(question: str, flat_nodes: list) -> list[dict]:
    """
    Ask the LLM which node_ids are most relevant to the question.
    Returns a list of node dicts from flat_nodes.
    """
    structure_text = build_structure_text(flat_nodes)

    system_prompt = (
        "You are a document navigation assistant. "
        "Given a hierarchical document index and a user question, "
        "identify the node_ids whose page content is most likely to contain "
        "the information needed to answer the question. "
        "Return ONLY a JSON array of node_id strings, e.g. [\"0001\", \"0009\"]. "
        "Select 3-6 of the most relevant nodes. Do not include any explanation."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Document Index:\n{structure_text}"
    )

    response = _client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        selected_ids = json.loads(raw)
    except json.JSONDecodeError:
        import re
        selected_ids = re.findall(r'"(\d{4})"', raw)

    id_set = set(selected_ids)
    relevant = [n for n in flat_nodes if n["node_id"] in id_set]
    print(f"[qa_system] Relevant nodes selected: {[n['node_id'] for n in relevant]}")
    return relevant


# ── Step C: Page extraction via Docling ─────────────────────────────────────

def _get_docling_doc(pdf_path: Path):
    """
    Parse the PDF with Docling and cache the result.

    Docling gives structured text + proper markdown tables, far richer than
    raw plain-text extraction. The parsed document is cached in memory so
    repeated calls within the same session do not re-parse.
    """
    key = str(pdf_path)
    if key not in _docling_cache:
        print(f"[qa_system] Parsing PDF with Docling: {pdf_path.name} (one-time, cached)…")
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        _docling_cache[key] = result.document
        print("[qa_system] Docling parsing complete.")
    return _docling_cache[key]


def extract_pages(pdf_path: Path, page_ranges: list[tuple[int, int]]) -> str:
    """
    Extract text from the given 1-based page ranges using Docling.

    Advantages:
    - Financial tables rendered as proper Markdown tables
    - Multi-column layouts correctly re-ordered
    - Section headers preserved as headings

    Returns a single string with clearly labelled per-page sections.
    """
    doc = _get_docling_doc(pdf_path)

    target_pages = set()
    for start, end in page_ranges:
        for p in range(int(start), int(end) + 1):
            target_pages.add(p)

    page_sections = []
    for page_no in sorted(target_pages):
        chunks = []
        for item, _ in doc.iterate_items(page_no=page_no):
            # TableItem → export as markdown table; everything else → .text
            if hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(doc)
                except TypeError:
                    text = item.export_to_markdown()
            else:
                text = getattr(item, "text", "") or ""
            if text.strip():
                chunks.append(text.strip())
        if chunks:
            page_sections.append(f"--- Page {page_no} ---\n" + "\n\n".join(chunks))

    return "\n\n".join(page_sections)


def get_page_ranges(relevant_nodes: list[dict]) -> list[tuple[int, int]]:
    """Deduplicate and sort page ranges from selected nodes."""
    ranges = []
    for n in relevant_nodes:
        s, e = n.get("start_index"), n.get("end_index")
        if s is not None and e is not None:
            ranges.append((int(s), int(e)))
    return sorted(set(ranges))


# ── Step D: Final answer ─────────────────────────────────────────────────────

def generate_answer(question: str, context: str) -> str:
    """
    Use Azure OpenAI to answer the question in structured bullet-point format.
    """
    system_prompt = (
        "You are a financial analyst assistant. "
        "Your ONLY source of information is the document excerpts provided below. "
        "STRICT GROUNDING RULES — violating any rule makes your answer invalid:\n"
        "1. Every factual claim MUST be directly stated in the provided excerpts. "
        "   Do NOT add facts, names, products, or context from your training knowledge.\n"
        "2. If information needed to answer the question is NOT present in the excerpts, "
        "   explicitly state: 'Not mentioned in the provided excerpts.'\n"
        "3. Do NOT infer, extrapolate, or elaborate beyond what is explicitly written.\n"
        "4. Cite the page number (e.g. [p.3]) for every specific figure or claim.\n\n"
        "FORMAT RULES:\n"
        "5. Start with a one-sentence headline summary.\n"
        "6. Use top-level bullet points (•) for each major theme or metric.\n"
        "7. Use indented sub-bullets (  ◦) for segment-level or supporting details.\n"
        "8. Always include exact numbers, percentages, and dollar amounts as written in the source.\n"
        "9. Do NOT write paragraphs — every piece of information must be a bullet.\n"
        "10. End with a '📌 Key Takeaway' bullet summarising the single most important finding."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Document Excerpts:\n{context}"
    )

    response = _client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


# ── Main entrypoint ──────────────────────────────────────────────────────────

def answer_question(
    question: str,
    pdf_path: Path,
    structure_json_path: Path,
) -> dict:
    """
    Full vectorless RAG pipeline.

    Returns:
        {
            "question": str,
            "relevant_nodes": list,
            "pages_retrieved": list,
            "answer": str,
        }
    """
    data = load_structure(structure_json_path)
    flat = flatten_nodes(data.get("structure", []))
    print(f"[qa_system] Loaded {len(flat)} nodes from index.")

    relevant = identify_relevant_nodes(question, flat)

    page_ranges = get_page_ranges(relevant)
    print(f"[qa_system] Fetching pages: {page_ranges}")
    context = extract_pages(pdf_path, page_ranges)

    print("[qa_system] Generating answer...")
    answer = generate_answer(question, context)

    return {
        "question": question,
        "relevant_nodes": [
            {"node_id": n["node_id"], "title": n["title"], "pages": f"{n['start_index']}–{n['end_index']}"}
            for n in relevant
        ],
        "pages_retrieved": page_ranges,
        "context": context,   # raw retrieved text — used by evaluation.py
        "answer": answer,
    }


def find_pdf(data_dir: Path) -> Path:
    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF in {data_dir}")
    return pdfs[0]


def find_structure_json(results_dir: Path) -> Path:
    jsons = list(results_dir.glob("*_structure.json"))
    if not jsons:
        raise FileNotFoundError(
            f"No structure JSON in {results_dir}. Run index_pdf.py first."
        )
    return jsons[0]


if __name__ == "__main__":
    pdf_path = find_pdf(DATA_DIR)
    json_path = find_structure_json(RESULTS_DIR)

    result = answer_question(DEFAULT_QUESTION, pdf_path, json_path)

    print("\n" + "=" * 60)
    print("QUESTION:")
    print(result["question"])
    print("\nRELEVANT SECTIONS USED:")
    for n in result["relevant_nodes"]:
        print(f"  [{n['node_id']}] {n['title']}  (pages {n['pages']})")
    print("\nANSWER:")
    print(result["answer"])
    print("=" * 60)
