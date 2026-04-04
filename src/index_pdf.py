"""
index_pdf.py
------------
Indexes a PDF using PageIndex (vectorless RAG) and saves the hierarchical
JSON structure to the results/ directory.

Uses Azure OpenAI (configured in .env) via LiteLLM's azure/ prefix.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Project root is two levels up from src/
ROOT_DIR = Path(__file__).parent.parent

# Ensure src/ is on the path so the vendored `pageindex` package is importable
_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Load environment variables from .env at project root
load_dotenv(ROOT_DIR / ".env")

# Map .env variables to LiteLLM Azure format
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_API_KEY", "")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_ENDPOINT", "")
os.environ["AZURE_API_VERSION"] = os.getenv("API_VERSION", "")

# Model string for LiteLLM Azure routing
AZURE_MODEL = f"azure/{os.getenv('DEPLOYMENT_NAME', 'gpt-4o')}"

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"


def find_pdf(data_dir: Path) -> Path:
    """Find the single PDF in the data directory."""
    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {data_dir}")
    if len(pdfs) > 1:
        print(f"Multiple PDFs found; using: {pdfs[0].name}")
    return pdfs[0]


def run_indexing(pdf_path: Path, output_dir: Path, model: str = AZURE_MODEL) -> dict:
    """
    Run PageIndex on a PDF and return the structured JSON result.

    Args:
        pdf_path:   Path to the PDF file.
        output_dir: Directory where the output JSON will be saved.
        model:      LiteLLM model string (e.g. "azure/gpt-4o").

    Returns:
        The PageIndex result dictionary.
    """
    from pageindex.page_index import page_index  # noqa: import after env setup

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}_structure.json"

    # Check for cached result
    if output_path.exists():
        print(f"[index_pdf] Cached result found: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[index_pdf] Indexing: {pdf_path.name}  model={model}")
    result = page_index(
        doc=str(pdf_path),
        model=model,
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_doc_description="yes",
        if_add_node_text="no",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[index_pdf] Saved → {output_path}")
    return result


def print_structure_tree(nodes: list, indent: int = 0) -> None:
    """Recursively print the document structure as an indented tree."""
    for node in nodes:
        prefix = "  " * indent + ("└─ " if indent else "")
        pages = f"[p{node.get('start_index', '?')}–{node.get('end_index', '?')}]"
        nid = node.get("node_id", "")
        print(f"{prefix}{nid}  {node.get('title', 'Untitled')}  {pages}")
        children = node.get("nodes") or node.get("children") or []
        if children:
            print_structure_tree(children, indent + 1)


if __name__ == "__main__":
    pdf_path = find_pdf(DATA_DIR)
    result = run_indexing(pdf_path, RESULTS_DIR)

    print("\n=== Document Structure Tree ===")
    print(f"Doc: {result.get('doc_name', pdf_path.name)}")
    print(f"Description: {result.get('doc_description', '')[:200]}...")
    print()
    structure = result.get("structure", [])
    print_structure_tree(structure)
    print(f"\nTotal top-level sections: {len(structure)}")
