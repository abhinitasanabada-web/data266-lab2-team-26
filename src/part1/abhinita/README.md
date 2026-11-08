# DATA 266 Lab 2 - Team #26

## Part 1 - Multimodal RAG over a read-only knowledge base

Notebook: `notebooks/p1_rag_v6.ipynb` (tag `p1-v1.0`). Parts 2 and 3 are not in this folder yet.

> This README describes **v6, the final version** (v4 and v5 are kept in `notebooks/` for comparison). The earlier v1 is kept in `v1/` with its own README ([`v1/README.md`](v1/README.md)).

Hosted models used: none. Everything runs locally on one GPU, with no API keys.

### Models
| Purpose | Model |
|---|---|
| Figure reading + answers | `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit nf4, bitsandbytes); `Qwen/Qwen2.5-VL-3B-Instruct` only as an automatic fallback |
| Text embeddings | `BAAI/bge-small-en-v1.5` |
| Reranker | `BAAI/bge-reranker-base` |

### How to run
1. **Colab:** Runtime > Change runtime type > GPU, then upload the competition zip next to the notebook.
   **Local GPU (RTX 4070 / A100):** create a Python env, put the competition files (or the zip) next to the notebook.
   **Kaggle:** attach the competition data, GPU on, Internet on for the pip cell.
2. **Restart & Run All.** The notebook installs its packages, builds the knowledge base, answers
   `QUESTIONS_PATH` (default `Questions.csv`), checks the format with `validate_submission.py`
   and writes `submission.csv`.
3. Golden-set evaluation (section 13) runs automatically when `eval/golden_set.csv` has rows.

Fits on one 16 GB GPU (the 7B model is loaded in 4-bit). Tested on a Colab T4 and an RTX 5090 (32 GB, CUDA 12.8 build of PyTorch);
the install cell checks that PyTorch supports the GPU and upgrades it if needed. The setup cell prints the GPU profile it chose
(`VLM_DTYPE`, `VLM_BATCH`). To reproduce another machine's run exactly, set those two environment
variables to the printed values and copy that machine's `work/kb_cache/` folder (cached figure readings).
Other optional variables: `QUESTIONS_PATH`, `DATA_DIR` (folder with the competition files), `VLM_MODEL`, `ANSWER_4BIT`, `FIG_TILES` (0 = skip the zoomed figure view), and the A/B switches listed under *Changes from v5 to v6*.
Every cell prints its start/end time and GPU memory (also written to `work/run_trace.log`), and the answering cell prints one line per question.

### Pipeline
PyMuPDF ingestion (text chunks, table words, figure crops) -> VLM figure reading in two views (whole chart + zoomed halves,
   majority vote on disagreements; cached)
-> fact store with precedence rules (revision > outturn > forecast, trust level, corrected figures)
-> ChromaDB dense + BM25 hybrid retrieval (RRF + exact-label boost + reranker)
-> planner (lookups, comparisons, changes, rankings, two-figure questions, follow-ups, session values, edits, false premises)
-> LangGraph answer graph -> guardrails (regex rails, NVIDIA NeMo Guardrails, number grounding, citation check)
-> JSON answer (<= 120 words).

### Design rules against hallucination
* Low-trust content (DOC3 Appendix C) is stored in ChromaDB with `trust_level=low` but excluded from
  retrieval; text that gives instructions to AI systems is dropped and treated as data, never followed.
* A figure or table named in the question is fetched by its exact label; a corrected figure replaces the original.
* Numbers come from the fact store or the retrieved evidence; citations and `figure_references`
  are built from the evidence actually used.
* Grounding rail: a generated sentence with a number that is not in the evidence, the facts, the question or a
  calculation is removed; citations that do not exist (or point to low-trust pages) are dropped.
* Prompt-injection, off-topic and harmful requests are refused; requests to change the documents are refused politely,
  and session-only values are used only inside their conversation.
* Values the user supplies live only in that conversation's memory, never in the index.
* Every question produces a row; unanswerable or false-premise questions get the matching status.

### Repository layout
```
notebooks/p1_rag_v6.ipynb   FINAL Part 1 notebook (executed, outputs saved) - self-contained, no local imports
notebooks/p1_rag_v5.ipynb   previous version (leaderboard 0.895)
notebooks/p1_rag_v4.ipynb   previous version (leaderboard 0.855)
eval/golden_set.csv         our 25 golden questions (written before tuning)
requirements.txt            pinned versions (written by the v6 notebook's last cell)
CONTRIBUTIONS.md            who did what + meeting recordings
AI_USAGE.md                 AI coding assistants used

v1/                         earlier version, kept for history (not used by v6)
  README.md                   the v1 README
  p1_rag_v1.ipynb             first notebook; imports the rag/ package
  p1_rag_v1_0_8_context.ipynb v1 variant with a longer retrieval context
  rag/                        v1 modules: ingest, figure reading, ChromaDB + BM25 index, retrieval, answering
  tool/                       v1 helper script that regenerates the v1 notebooks from rag/*.py
```
Data, model weights, `work/` caches, `submission.csv` and secrets are not committed (see `.gitignore`).

### Version history
| Version | What changed |
|---|---|
| v1 | Modular `rag/` package: hybrid retrieval (dense + BM25 + RRF), the VLM reads figures, the LLM writes every answer. |
| v4 | Single notebook; adds a deterministic fact store (table parser, prose and figure facts) with precedence rules, a question planner, a reranker, LangGraph + NeMo Guardrails, and a GPU profile for T4 / RTX 4070 / A100. Leaderboard 0.855. |
| v5 | Public leaderboard 0.895. Fixes bar-chart readings (a parsing bug dropped every bar value), two-view figure reading with a vote, clean margin captions for DOC1 figures, value snapping to the same document's tables, two-figure / change / ranking questions, tolerant wording (Fig., USA, end of 2010, ...), Okun's law from the named document, layered guardrails with a number-grounding check, and fallbacks so a run never crashes or runs out of time. |
| v6 (final) | Public leaderboard 0.899. Same pipeline as v5 with a more robust reader for the model's chart transcriptions, safer figure voting, data-folder detection on any machine and optional A/B switches. See *Changes from v5 to v6* below. |

### Changes from v5 to v6
Measured on the public leaderboard (practice only; the graded score comes from the hidden question set).

| Run | Public score | What changed |
|---|---|---|
| v5 (RTX 5090) | 0.895 | baseline for this round |
| v6 attempt | 0.870 | citation pages chosen by word overlap with the answer, and explanations cite tables only for "how large" questions. Both **reverted**. |
| v6 attempt | 0.888 | verify-and-revise loop: a second LLM pass checks every "X% in YEAR" claim against the evidence. Did not help; now **off by default** (`VERIFY_YEARS=1` turns it on). |
| **v6 final** | **0.899** | the fixes below, all optional switches off |

What improved from v5 to v6:
1. **Chart transcriptions parsed robustly.** On the GPU the model often wrote its chart readings in shapes the v5 parser skipped:
   markdown headings (`**Data:**`, `### DATA`), values grouped under a series heading, colour names instead of series names,
   and lines like `CPI (headline): 2009 ...`. v6 finds the sections anywhere on a line, maps colours to series through the
   chart's legend and reads grouped rows, so those charts now produce facts instead of being lost.
2. **Safer figure voting.** The vote between the whole-chart view and the zoomed view now keys every value by series *and*
   variable, so two series of the same chart (for example headline vs core inflation) can no longer swap values.
   A chart transcribed point by point (6+ "printed" values) keeps a value as printed only where the zoomed view agrees and the
   point is the last, lowest or highest one, so read-off-the-line estimates are not cited as printed labels.
3. **Explanation answers kept on the v5 rule** (cite the pages the LLM actually used, tables allowed); the two alternatives
   tried above scored lower.
4. **Runs anywhere.** The data folder is found through `DATA_DIR`, the current folder, Kaggle input, Colab and Google Drive;
   on Colab the notebook asks for the competition zip if it is missing, instead of failing on `kb_manifest.json`.
5. **Optional A/B switches** in the "machine settings" cell, all off by default: `EXPLAIN_TABLES` (`always` / `magnitude` / `never`),
   `EXPLAIN_PAGES` (1 or 2 cited pages for explanations), `SHORT_RETRY`, `SAMPLE_SHAPES`, `VERIFY_YEARS`.
   The submitted v6 run uses none of them.

No rule in v6 refers to a specific question: every change applies to all questions of its type.

Only v6 is the graded submission. v1 is kept to show how the system evolved.
