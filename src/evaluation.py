"""
evaluation.py
-------------
GEval-based comparison of PageIndex RAG vs Section-Based Vector RAG.

Since there is no gold-standard answer, the full PDF text (extracted via
Docling) is supplied as `context` so GEval can assess completeness and
faithfulness against the actual source document.

Five custom GEval metrics:
  1. Answer Relevancy   — does the answer directly address the question?
  2. Faithfulness       — every claim grounded in the source document?
  3. Completeness       — key facts from the PDF covered?
  4. Conciseness        — appropriately detailed, not verbose or sparse?
  5. Coherence          — logically structured and easy to follow?

Each metric produces a score (0–1) AND a detailed reasoning string.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

load_dotenv(ROOT_DIR / ".env")

from openai import AzureOpenAI                          # noqa: E402
from deepeval.metrics import GEval                      # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: E402
from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402


# ── Azure judge model for DeepEval ───────────────────────────────────────────

class _AzureJudge(DeepEvalBaseLLM):
    """Wraps Azure OpenAI so DeepEval can use it as its evaluation LLM."""

    def __init__(self):
        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("API_VERSION"),
        )
        self._deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

    def load_model(self):
        return self._client

    def generate(self, prompt: str) -> str:
        # Accepts only `prompt` — no *args/**kwargs.
        # DeepEval calls generate(prompt, schema=X) for structured steps;
        # the missing `schema` kwarg raises TypeError which DeepEval catches
        # and falls back to generate(prompt), receiving a plain string.
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._deployment


# ── GEval metric definitions ──────────────────────────────────────────────────

_METRICS_CONFIG = [
    {
        "name": "Answer Relevancy",
        "criteria": (
            "Assess whether the answer directly and completely addresses the question asked. "
            "A highly relevant answer stays focused on the question without introducing "
            "unrelated information. Penalise answers that are off-topic or only partially "
            "address the question."
        ),
        "params": [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    },
    {
        "name": "Faithfulness",
        "criteria": (
            "Assess whether every factual claim made in the answer is explicitly supported "
            "by the provided document context. An unfaithful answer contains hallucinations, "
            "fabricated numbers, or statements that cannot be verified from the context. "
            "Score 1 only if all claims are traceable to the context."
        ),
        "params": [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
    },
    {
        "name": "Completeness",
        "criteria": (
            "Using the full document content provided in 'context', assess whether the answer "
            "covers all key facts and figures that are relevant to the question. "
            "Penalise answers that omit important metrics (e.g. revenue numbers, segment "
            "breakdowns, year-over-year comparisons) that are present in the document."
        ),
        "params": [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
    },
    {
        "name": "Conciseness",
        "criteria": (
            "Assess whether the answer is appropriately concise: it must contain all necessary "
            "information without excessive repetition or padding, but must not be so brief that "
            "important details are omitted. Penalise both over-verbose and under-detailed answers."
        ),
        "params": [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    },
    {
        "name": "Coherence",
        "criteria": (
            "Assess whether the answer is logically structured, flows naturally, and is easy "
            "to understand. The answer should present information in a clear, ordered manner "
            "with consistent reasoning. Penalise disorganised, contradictory, or hard-to-follow "
            "answers."
        ),
        "params": [
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    },
]


# ── Full PDF context ──────────────────────────────────────────────────────────

def get_full_pdf_context(pdf_path: Path, max_chars: int = 40_000) -> str:
    """
    Extract the full PDF text with Docling (reuses the shared cache from qa_system).
    Truncated to `max_chars` to respect LLM context limits.
    """
    from qa_system import extract_pages                 # noqa — uses Docling cache
    from pageindex.utils import get_number_of_pages     # noqa

    n = get_number_of_pages(str(pdf_path))
    # Extract all pages in one call
    full_text = extract_pages(pdf_path, [(1, n)])
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[… truncated …]"
    return full_text


# ── Core evaluation ───────────────────────────────────────────────────────────

def _run_metric(
    metric_cfg: dict,
    question: str,
    answer: str,
    retrieval_context: list[str],
    full_pdf_context: list[str],
    judge: _AzureJudge,
) -> dict:
    """
    Run a single GEval metric and return {name, score, reason}.
    """
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context,
        context=full_pdf_context,
    )

    metric = GEval(
        name=metric_cfg["name"],
        criteria=metric_cfg["criteria"],
        evaluation_params=metric_cfg["params"],
        model=judge,
        async_mode=False,   # synchronous so we can collect results one by one
        verbose_mode=False,
    )
    metric.measure(test_case)

    return {
        "metric": metric_cfg["name"],
        "score": round(metric.score, 3),
        "reason": metric.reason or "No reasoning provided.",
    }


def evaluate_answer(
    question: str,
    answer: str,
    retrieval_context: list[str],
    full_pdf_text: str,
    judge: _AzureJudge,
    label: str = "answer",
) -> list[dict]:
    """
    Run all 5 GEval metrics for a single answer.

    Args:
        retrieval_context: list of retrieved chunk texts (what the RAG actually used)
        full_pdf_text:     entire PDF text (used as 'context' for Completeness)
    """
    print(f"\n[evaluation] Evaluating '{label}' across 5 metrics…")
    results = []
    for cfg in _METRICS_CONFIG:
        print(f"  • {cfg['name']}…", end=" ", flush=True)
        r = _run_metric(
            cfg,
            question=question,
            answer=answer,
            retrieval_context=retrieval_context,
            full_pdf_context=[full_pdf_text],
            judge=judge,
        )
        print(f"score={r['score']}")
        results.append(r)
    return results


# ── Main comparison ───────────────────────────────────────────────────────────

def compare_answers(
    question: str,
    pageindex_result: dict,
    vector_result: dict,
    pdf_path: Path,
) -> dict:
    """
    Compare PageIndex RAG vs Vector RAG using 5 GEval metrics.

    Args:
        pageindex_result: output of qa_system.answer_question()
        vector_result:    output of vector_rag.vector_rag_answer()
        pdf_path:         path to the PDF (for full-text extraction)

    Returns:
        {
            "question": str,
            "pageindex": {
                "answer": str,
                "scores": [{"metric", "score", "reason"}, …]
            },
            "vector_rag": {
                "answer": str,
                "scores": [{"metric", "score", "reason"}, …]
            },
            "summary": [{"metric", "pageindex_score", "vector_score", "winner"}, …]
        }
    """
    judge = _AzureJudge()

    print("[evaluation] Extracting full PDF context for Completeness metric…")
    full_pdf_text = get_full_pdf_context(pdf_path)
    print(f"[evaluation] Full PDF context: {len(full_pdf_text):,} chars.")

    # Retrieval context for each approach (list of strings — one per chunk/page)
    pi_retrieval = [pageindex_result.get("context", pageindex_result.get("answer", ""))]
    vr_retrieval = [chunk["text"] if isinstance(chunk, dict) else chunk
                    for chunk in [vector_result.get("context", vector_result.get("answer", ""))]]

    # Evaluate both
    pi_scores = evaluate_answer(
        question=question,
        answer=pageindex_result["answer"],
        retrieval_context=pi_retrieval,
        full_pdf_text=full_pdf_text,
        judge=judge,
        label="PageIndex RAG",
    )
    vr_scores = evaluate_answer(
        question=question,
        answer=vector_result["answer"],
        retrieval_context=vr_retrieval,
        full_pdf_text=full_pdf_text,
        judge=judge,
        label="Vector RAG",
    )

    # Build summary
    summary = []
    for pi, vr in zip(pi_scores, vr_scores):
        assert pi["metric"] == vr["metric"]
        if pi["score"] > vr["score"]:
            winner = "PageIndex RAG"
        elif vr["score"] > pi["score"]:
            winner = "Vector RAG"
        else:
            winner = "Tie"
        summary.append({
            "metric": pi["metric"],
            "pageindex_score": pi["score"],
            "vector_score": vr["score"],
            "winner": winner,
        })

    return {
        "question": question,
        "pageindex": {"answer": pageindex_result["answer"], "scores": pi_scores},
        "vector_rag": {"answer": vector_result["answer"], "scores": vr_scores},
        "summary": summary,
    }


if __name__ == "__main__":
    from index_pdf import find_pdf, DATA_DIR, RESULTS_DIR
    from qa_system import load_structure, find_structure_json, answer_question
    from qa_system import DEFAULT_QUESTION as QUESTION
    from vector_rag import vector_rag_answer

    pdf_path = find_pdf(DATA_DIR)
    structure = load_structure(find_structure_json(RESULTS_DIR))
    json_path = find_structure_json(RESULTS_DIR)

    pi_result = answer_question(QUESTION, pdf_path, json_path)
    vr_result = vector_rag_answer(QUESTION, pdf_path, structure)

    # Attach context string to pi_result for evaluation
    from qa_system import extract_pages, get_page_ranges, identify_relevant_nodes, flatten_nodes
    flat = flatten_nodes(structure.get("structure", []))
    relevant = identify_relevant_nodes(QUESTION, flat)
    pi_result["context"] = extract_pages(pdf_path, get_page_ranges(relevant))

    comparison = compare_answers(QUESTION, pi_result, vr_result, pdf_path)

    print("\n" + "=" * 60)
    print("GEVAL COMPARISON SUMMARY")
    print("=" * 60)
    for row in comparison["summary"]:
        print(f"  {row['metric']:<22} PI={row['pageindex_score']}  VR={row['vector_score']}  → {row['winner']}")
