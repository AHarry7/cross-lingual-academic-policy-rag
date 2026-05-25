"""
eval/evaluate.py
────────────────
Full three-pipeline ablation study using RAGAS.

Pipelines evaluated:
  1. Baseline  — direct cross-lingual retrieval, no query normalization
  2. Advanced  — LLM query normalization + dense ChromaDB retrieval
  3. Hybrid    — LLM query normalization + BM25 + ChromaDB + RRF fusion

Metrics (all four require an LLM judge):
  - Context Precision   : Are retrieved chunks relevant to the question?
  - Faithfulness        : Is the answer grounded in retrieved context?
  - Answer Relevancy    : Does the answer address the question asked?
  - Answer Correctness  : Does the answer match the ground truth?

Outputs (saved to eval/results/):
  - {pipeline}_results.json   : per-question scores + metadata
  - {pipeline}_results.csv    : per-question scores (for Excel/thesis tables)
  - ablation_summary.csv      : 3-pipeline mean scores side by side

Usage:
  python -m eval.evaluate
  (run from the project root: roman-urdu-rag/)
"""

import os
import sys
import json
import time
import csv
from datetime import datetime
from dotenv import load_dotenv

# ── Path fix so src.* imports work when running from project root ────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

# ── RAGAS imports ────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from datasets import Dataset

# ── LLM judge (Groq via OpenAI-compatible client) ────────────────────────────
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── Project pipelines ────────────────────────────────────────────────────────
from src.generator import get_rag_chain
from src.advanced_generator import advanced_rag_query
from src.hybrid_generator import hybrid_rag_query
from src.retriever import build_or_load_vector_store
from src.hybrid_retriever import HybridRetriever

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_DIR         = os.path.join(os.path.dirname(__file__), "results")
METRICS             = [context_precision, faithfulness, answer_relevancy, answer_correctness]
METRIC_NAMES        = ["context_precision", "faithfulness", "answer_relevancy", "answer_correctness"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_golden_dataset() -> list[dict]:
    print(f"Loading golden dataset from {GOLDEN_DATASET_PATH}...")
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} questions.\n")
    return data


def setup_ragas_judge() -> tuple:
    """
    RAGAS needs an LLM and embedding model to judge answers.

    LLM    : Groq llama-3.1-8b-instant (free, fast)
    Embeds : paraphrase-multilingual-MiniLM-L12-v2 (same model as retrieval)

    Using the same HuggingFace embedding model as ChromaDB keeps the entire
    stack free, open-source, and consistent — no OpenAI key required.
    All 4 metrics including answer_relevancy will run.
    """
    print("Setting up RAGAS judge...")
    print("  LLM       : Groq llama-3.1-8b-instant")
    print("  Embeddings: paraphrase-multilingual-MiniLM-L12-v2 (HuggingFace)")

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    ragas_llm = LangchainLLMWrapper(llm)

    # Reuse the same multilingual embedding model used for ChromaDB retrieval.
    # This is intentional — keeps the evaluation stack fully free and consistent.
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    print("  Judge ready. All 4 metrics will run.\n")
    return ragas_llm, ragas_embeddings


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNERS
# Each returns a list of dicts with: question, answer, contexts, ground_truth
# plus the original metadata (id, tier, topic) for per-tier analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(golden_data: list[dict]) -> list[dict]:
    """Run all questions through the Baseline pipeline."""
    print("\n" + "="*60)
    print("RUNNING PIPELINE 1/3: BASELINE")
    print("="*60)

    chain = get_rag_chain()
    db    = build_or_load_vector_store()
    results = []

    for i, item in enumerate(golden_data):
        print(f"  [{i+1:02d}/{len(golden_data)}] {item['question'][:60]}...")
        try:
            response = chain.invoke({"input": item["question"]})
            answer   = response["answer"]

            # Re-run retrieval separately to capture contexts for RAGAS
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs      = retriever.invoke(item["question"])
            contexts  = [doc.page_content for doc in docs]

        except Exception as e:
            print(f"    ⚠ Error on question {item['id']}: {e}")
            answer   = "ERROR"
            contexts = []

        results.append({
            "id":           item["id"],
            "tier":         item["tier"],
            "topic":        item["topic"],
            "question":     item["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })

        # Small delay to avoid Groq rate limits
        time.sleep(1.5)

    print(f"  ✓ Baseline complete — {len(results)} questions processed.\n")
    return results


def run_advanced(golden_data: list[dict]) -> list[dict]:
    """Run all questions through the Advanced pipeline."""
    print("\n" + "="*60)
    print("RUNNING PIPELINE 2/3: ADVANCED")
    print("="*60)

    results = []

    for i, item in enumerate(golden_data):
        print(f"  [{i+1:02d}/{len(golden_data)}] {item['question'][:60]}...")
        try:
            result   = advanced_rag_query(item["question"])
            answer   = result["answer"]

            # Re-retrieve using the normalized English query for contexts
            from src.retriever import build_or_load_vector_store as load_db
            db        = load_db()
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs      = retriever.invoke(result["expanded_query"])
            contexts  = [doc.page_content for doc in docs]

        except Exception as e:
            print(f"    ⚠ Error on question {item['id']}: {e}")
            answer   = "ERROR"
            contexts = []

        results.append({
            "id":           item["id"],
            "tier":         item["tier"],
            "topic":        item["topic"],
            "question":     item["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })

        time.sleep(1.5)

    print(f"  ✓ Advanced complete — {len(results)} questions processed.\n")
    return results


def run_hybrid(golden_data: list[dict]) -> list[dict]:
    """Run all questions through the Hybrid pipeline."""
    print("\n" + "="*60)
    print("RUNNING PIPELINE 3/3: HYBRID")
    print("="*60)

    hybrid_retriever = HybridRetriever()
    results = []

    for i, item in enumerate(golden_data):
        print(f"  [{i+1:02d}/{len(golden_data)}] {item['question'][:60]}...")
        try:
            result   = hybrid_rag_query(item["question"])
            answer   = result["answer"]

            # Re-retrieve using hybrid for contexts
            docs     = hybrid_retriever.retrieve(result["expanded_query"], k=3)
            contexts = [doc.page_content for doc in docs]

        except Exception as e:
            print(f"    ⚠ Error on question {item['id']}: {e}")
            answer   = "ERROR"
            contexts = []

        results.append({
            "id":           item["id"],
            "tier":         item["tier"],
            "topic":        item["topic"],
            "question":     item["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })

        time.sleep(1.5)

    print(f"  ✓ Hybrid complete — {len(results)} questions processed.\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline(
    pipeline_name: str,
    results: list[dict],
    ragas_llm,
    ragas_embeddings,
) -> dict:
    """
    Runs RAGAS evaluation on a pipeline's results.
    Returns a dict of mean scores per metric.
    Also saves per-question scores to JSON and CSV.
    """
    print(f"\nEvaluating {pipeline_name} with RAGAS...")

    # Build RAGAS Dataset — only the 4 required columns
    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    # Select metrics — skip answer_relevancy if no embeddings available
    active_metrics = METRICS if ragas_embeddings else [
        m for m in METRICS if m != answer_relevancy
    ]
    active_names = METRIC_NAMES if ragas_embeddings else [
        n for n in METRIC_NAMES if n != "answer_relevancy"
    ]

    # Configure each metric with the judge LLM and embeddings
    for metric in active_metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings") and ragas_embeddings:
            metric.embeddings = ragas_embeddings

    # Run evaluation
    eval_result = evaluate(dataset, metrics=active_metrics)
    scores_df   = eval_result.to_pandas()

    # ── Attach metadata columns for per-tier analysis ─────────────────────
    scores_df.insert(0, "id",    [r["id"]    for r in results])
    scores_df.insert(1, "tier",  [r["tier"]  for r in results])
    scores_df.insert(2, "topic", [r["topic"] for r in results])

    # ── Save JSON ──────────────────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, f"{pipeline_name.lower()}_results.json")
    records   = scores_df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Saved JSON → {json_path}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, f"{pipeline_name.lower()}_results.csv")
    scores_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Saved CSV  → {csv_path}")

    # ── Compute mean scores ────────────────────────────────────────────────
    mean_scores = {}
    for name in active_names:
        if name in scores_df.columns:
            mean_scores[name] = round(float(scores_df[name].mean()), 4)
        else:
            mean_scores[name] = None

    # Fill missing metrics as None
    for name in METRIC_NAMES:
        if name not in mean_scores:
            mean_scores[name] = None

    return mean_scores


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(all_scores: dict[str, dict]):
    """Prints a formatted 3-pipeline comparison table to the terminal."""

    pipelines = list(all_scores.keys())
    col_w     = 12

    # Header
    print("\n")
    print("╔" + "═"*24 + "╦" + ("═"*col_w + "╦") * (len(pipelines)-1) + "═"*col_w + "╗")
    header = f"║ {'Metric':<22} ║"
    for p in pipelines:
        header += f" {p:^{col_w-2}} ║"
    print(header)
    print("╠" + "═"*24 + "╬" + ("═"*col_w + "╬") * (len(pipelines)-1) + "═"*col_w + "╣")

    # Rows
    display_names = {
        "context_precision":  "Context Precision",
        "faithfulness":       "Faithfulness",
        "answer_relevancy":   "Answer Relevancy",
        "answer_correctness": "Answer Correctness",
    }

    for metric_key, metric_label in display_names.items():
        row = f"║ {metric_label:<22} ║"
        for p in pipelines:
            val = all_scores[p].get(metric_key)
            cell = f"{val:.4f}" if val is not None else "  N/A  "
            row += f" {cell:^{col_w-2}} ║"
        print(row)

    print("╚" + "═"*24 + "╩" + ("═"*col_w + "╩") * (len(pipelines)-1) + "═"*col_w + "╝")
    print()


def save_ablation_summary(all_scores: dict[str, dict]):
    """Saves the 3-pipeline mean score comparison to a CSV."""
    summary_path = os.path.join(RESULTS_DIR, "ablation_summary.csv")

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow(["Metric"] + list(all_scores.keys()))

        # One row per metric
        for metric_key in METRIC_NAMES:
            row = [metric_key]
            for pipeline in all_scores:
                val = all_scores[pipeline].get(metric_key)
                row.append(f"{val:.4f}" if val is not None else "N/A")
            writer.writerow(row)

    print(f"  Saved ablation summary → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start_time = datetime.now()

    print("\n" + "█"*60)
    print("  CROSS-LINGUAL RAG — ABLATION STUDY (RAGAS)")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("█"*60)

    ensure_results_dir()
    golden_data              = load_golden_dataset()
    ragas_llm, ragas_embeds  = setup_ragas_judge()

    # ── Step 1: Collect pipeline outputs ─────────────────────────────────────
    baseline_results = run_baseline(golden_data)
    advanced_results = run_advanced(golden_data)
    hybrid_results   = run_hybrid(golden_data)

    # ── Step 2: Evaluate each pipeline with RAGAS ─────────────────────────
    all_scores = {}
    all_scores["Baseline"] = evaluate_pipeline(
        "Baseline", baseline_results, ragas_llm, ragas_embeds
    )
    all_scores["Advanced"] = evaluate_pipeline(
        "Advanced", advanced_results, ragas_llm, ragas_embeds
    )
    all_scores["Hybrid"] = evaluate_pipeline(
        "Hybrid", hybrid_results, ragas_llm, ragas_embeds
    )

    # ── Step 3: Print terminal summary and save ablation CSV ──────────────
    print("\n" + "█"*60)
    print("  ABLATION STUDY RESULTS")
    print("█"*60)
    print_summary_table(all_scores)
    save_ablation_summary(all_scores)

    elapsed = (datetime.now() - start_time).seconds // 60
    print(f"\n  Total time: ~{elapsed} minutes")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()