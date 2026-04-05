# PageIndex — Vectorless RAG + Vector RAG with GEval Evaluation

A complete financial document Q&A system combining two RAG approaches and evaluated using GEval — built on top of [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex).

---

## What It Does

| Approach | How retrieval works | Text extraction |
|---|---|---|
| **PageIndex RAG** | LLM navigates a hierarchical structure tree | Docling |
| **Section-Based Vector RAG** | FAISS cosine search over sentence embeddings | Docling |

Both approaches answer the same question and are compared using **5 GEval metrics** (Answer Relevancy, Faithfulness, Completeness, Conciseness, Coherence) — no gold-standard answer required.

---

## Project Structure

```
Page_Index/
├── .env                          # Azure OpenAI credentials (not committed)
├── requirements.txt
├── data/
│   └── *.pdf                     # Put your PDF here (see Data section below)
├── results/
│   ├── *_structure.json          # PageIndex hierarchical index (auto-generated)
│   ├── full_comparison.json      # GEval comparison results
│   └── geval_comparison.png      # Bar chart
├── src/
│   ├── pageindex/                # Vendored PageIndex library (VectifyAI/PageIndex)
│   ├── index_pdf.py              # Step 1 — index PDF → hierarchical JSON
│   ├── qa_system.py              # Step 2 — PageIndex RAG Q&A
│   ├── vector_rag.py             # Step 3 — Section-based Vector RAG (FAISS)
│   └── evaluation.py             # Step 4 — GEval 5-metric comparison
└── notebooks/
    └── demo.ipynb                # Full demo — run this
```

---

## Data — Getting the PDF

This project is designed for financial PDF reports. The benchmark used to validate it is **FinanceBench** ([paper](https://arxiv.org/pdf/2311.11944)), the same dataset used by [VectifyAI/Mafin2.5](https://github.com/VectifyAI/Mafin2.5-FinanceBench) to achieve **98.7% accuracy**.

### Option 1 — Download from FinanceBench (recommended)

The FinanceBench PDF collection is hosted at [`patronus-ai/financebench`](https://github.com/patronus-ai/financebench/tree/main/pdfs).

```bash
# Clone only the pdfs folder (sparse checkout — avoids downloading the full repo)
git clone --no-checkout https://github.com/patronus-ai/financebench.git tmp_fb
cd tmp_fb
git sparse-checkout init --cone
git sparse-checkout set pdfs
git checkout main

# Copy a PDF into the data/ folder
cp pdfs/AMZN_2022_10K.pdf ../data/
cd .. && rm -rf tmp_fb
```

Or download a single PDF directly (example — Amazon 2022 10-K):
```bash
curl -L -o data/AMZN_2022_10K.pdf \
  "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/AMZN_2022_10K.pdf"
```

Browse all available PDFs at:
> https://github.com/patronus-ai/financebench/tree/main/pdfs

### Option 2 — Use your own PDF

Drop any financial PDF into the `data/` folder. The pipeline auto-discovers it:
```
data/
└── your-report.pdf
```

> **Note:** Only one PDF should be in `data/` at a time.

---

## Setup

### 1. Configure Azure OpenAI

Create a `.env` file in the project root:

```env
AZURE_API_KEY=your_api_key
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
DEPLOYMENT_NAME=gpt-4o
API_VERSION=2025-01-01-preview
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

Or run each step individually from the project root:

```bash
# Step 1 — Build the hierarchical index
python src/index_pdf.py

# Step 2 — PageIndex RAG answer
python src/qa_system.py

# Step 3 — Vector RAG answer
python src/vector_rag.py

# Step 4 — GEval comparison
python src/evaluation.py
```

---

## Pipeline Overview

```
PDF
 │
 ├─► [index_pdf.py]
 │     PageIndex (LLM-based)
 │     └─► results/*_structure.json   ← hierarchical section tree
 │
 ├─► [qa_system.py]  ← PageIndex RAG
 │     LLM reads tree → selects nodes → Docling extracts pages → Azure OpenAI answers
 │
 ├─► [vector_rag.py]  ← Section-Based Vector RAG
 │     Docling extracts sections → sentence-transformers embeds → FAISS → Azure OpenAI answers
 │
 └─► [evaluation.py]  ← GEval (5 metrics, no gold standard)
       Full PDF via Docling → DeepEval GEval → scores + reasoning per metric
```

---

## GEval Metrics

| Metric | What it evaluates | Needs gold standard? |
|---|---|---|
| Answer Relevancy | Does the answer address the question? | No |
| Faithfulness | Every claim grounded in retrieved content? | No |
| Completeness | Key facts from the full PDF covered? | No (uses full PDF) |
| Conciseness | Appropriately detailed, not verbose? | No |
| Coherence | Logically structured and clear? | No |

Judge LLM: **Azure OpenAI gpt-4o**  
Reference: **Full PDF text extracted via Docling** (up to 40 000 characters)

---

## Dependencies

| Package | Purpose |
|---|---|
| `docling` | PDF text extraction (tables as markdown, multi-column layout) |
| `litellm` | LLM routing for PageIndex internals (Azure/OpenAI) |
| `openai` | Azure OpenAI client for Q&A and evaluation |
| `faiss-cpu` | In-memory vector index |
| `sentence-transformers` | Local embeddings (`all-MiniLM-L6-v2`) |
| `GEval` | LLM-as-judge evaluation metrics |

---

## References

- [PageIndex — VectifyAI](https://github.com/VectifyAI/PageIndex)
- [Mafin2.5 FinanceBench Results](https://github.com/VectifyAI/Mafin2.5-FinanceBench) — 98.7% accuracy benchmark
- [FinanceBench Dataset — Patronus AI](https://github.com/patronus-ai/financebench)
- [FinanceBench Paper](https://arxiv.org/pdf/2311.11944)
- [Docling — IBM](https://github.com/DS4SD/docling)
- [DeepEval](https://github.com/confident-ai/deepeval)
