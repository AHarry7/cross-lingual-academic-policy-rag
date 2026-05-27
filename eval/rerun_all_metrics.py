"""
eval/run_full_evaluation.py
────────────────────────────
Single combined script:
  Step 1 — Runs all 3 pipelines on 10 representative questions
  Step 2 — Evaluates with RAGAS (strictness=1 for Groq compatibility)
  Step 3 — Saves JSON, CSV, and ablation_summary.csv
  Step 4 — Prints final summary table

10-question subset covers all 4 tiers:
  Clean (2), Misspelled (2), Code-mixed (2), Keyword-exact (4)

Usage (from project root):
  python -m eval.run_full_evaluation
"""

import os
import sys
import json
import csv
import time
import math
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# ── Project pipelines ────────────────────────────────────────────────────────
from src.generator import get_rag_chain
from src.retriever import build_or_load_vector_store
from src.advanced_generator import advanced_rag_query
from src.hybrid_generator import hybrid_rag_query
from src.hybrid_retriever import HybridRetriever

# ── RAGAS ────────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from ragas.run_config import RunConfig
from datasets import Dataset

# ── LLM + Embeddings ─────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_DIR         = os.path.join(os.path.dirname(__file__), "results")
METRIC_NAMES        = ["context_precision", "faithfulness", "answer_relevancy", "answer_correctness"]
ALL_METRICS         = [context_precision, faithfulness, answer_relevancy, answer_correctness]

# Representative 10-question subset — covers all 4 tiers
EVAL_SUBSET_IDS = {1, 3, 8, 9, 14, 15, 22, 24, 26, 27}

# Delay between pipeline calls to avoid Groq rate limits
CALL_DELAY_SECONDS = 2


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    subset = [r for r in data if r["id"] in EVAL_SUBSET_IDS]
    print(f"  Loaded {len(subset)}/28 questions (representative subset).")
    return subset


def setup_ragas_judge():
    print("Setting up RAGAS judge...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    ragas_llm = LangchainLLMWrapper(llm)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    print("  Judge ready.\n")
    return ragas_llm, ragas_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PIPELINE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(golden_data: list[dict]) -> list[dict]:
    print("\n" + "="*60)
    print("  PIPELINE 1/3: BASELINE")
    print("="*60)

    chain = get_rag_chain()
    db    = build_or_load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 5})
    results = []

    for i, item in enumerate(golden_data):
        print(f"  [{i+1:02d}/{len(golden_data)}] {item['question'][:65]}...")
        try:
            response = chain.invoke({"input": item["question"]})
            answer   = response["answer"]
            docs     = retriever.invoke(item["question"])
            contexts = [doc.page_content for doc in docs]
        except Exception as e:
            print(f"    Error: {e}")
            answer, contexts = "ERROR", []

        results.append({
            "id":           item["id"],
            "tier":         item["tier"],
            "topic":        item["topic"],
            "question":     item["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })
        time.sleep(CALL_DELAY_SECONDS)

    print(f"  Baseline complete — {len(results)} questions.\n")
    return results


def run_advanced(golden_data: list[dict]) -> list[dict]:
    print("\n" + "="*60)
    print("  PIPELINE 2/3: ADVANCED")
    print("="*60)

    results = []

    for i, item in enumerate(golden_data):
        print(f"  [{i+1:02d}/{len(golden_data)}] {item['question'][:65]}...")
        try:
            result   = advanced_rag_query(item["question"])
            answer   = result["answer"]
            db       = build_or_load_vector_store()
            retriever = db.as_retriever(search_kwargs={"k": 5})
            docs     = retriever.invoke(result["expanded_query"])
            contexts = [doc.page_content for doc in docs]
        except Exception as e:
            print(f"    Error: {e}")
            answer, contexts = "ERROR", []

        results.append({
            "id":             item["id"],
            "tier":           item["tier"],
            "topic":          item["topic"],
            "question":       item["question"],
            "expanded_query": result.get("expanded_query", ""),
            "answer":         answer,
            "contexts":       contexts,
            "ground_truth":   item["ground_truth"],
        })
        time.sleep(CALL_DELAY_SECONDS)

    print(f"  Advanced complete — {len(results)} questions.\n")
    return results


def run_hybrid(golden_data: list[dict]) -> list[dict]:
    print("\n" + "="*60)
    print("  PIPELINE 3/3: HYBRID")
    print("="*60)

    hybrid_retriever = HybridRetriever()
    results = []

    for i, item in enumerate(golden_data):
        print(f"  [{i+1:02d}/{len(golden_data)}] {item['question'][:65]}...")
        try:
            result   = hybrid_rag_query(item["question"])
            answer   = result["answer"]
            docs     = hybrid_retriever.retrieve(result["expanded_query"], k=5)
            contexts = [doc.page_content for doc in docs]
        except Exception as e:
            print(f"    Error: {e}")
            answer, contexts = "ERROR", []

        results.append({
            "id":             item["id"],
            "tier":           item["tier"],
            "topic":          item["topic"],
            "question":       item["question"],
            "expanded_query": result.get("expanded_query", ""),
            "answer":         answer,
            "contexts":       contexts,
            "ground_truth":   item["ground_truth"],
        })
        time.sleep(CALL_DELAY_SECONDS)

    print(f"  Hybrid complete — {len(results)} questions.\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — RAGAS EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline(
    pipeline_name: str,
    results: list[dict],
    ragas_llm,
    ragas_embeddings,
) -> dict:
    print(f"\nEvaluating {pipeline_name} with RAGAS...")

    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    # strictness=1 — critical for Groq: prevents n>1 generation requests
    # Without this, RAGAS requests n=3 which Groq rejects with BadRequestError
    context_precision.strictness  = 1
    faithfulness.strictness       = 1
    answer_relevancy.strictness   = 1
    answer_correctness.strictness = 1

    for metric in ALL_METRICS:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings") and ragas_embeddings:
            metric.embeddings = ragas_embeddings

    run_config = RunConfig(
        max_workers=1,   # one call at a time — no rate limit bursts
        max_retries=5,
        timeout=180,
        max_wait=60,
    )

    eval_result = evaluate(dataset, metrics=ALL_METRICS, run_config=run_config)
    scores_df   = eval_result.to_pandas()

    # Detect RAGAS output column name (may rename question → user_input)
    df_q_col = "user_input" if "user_input" in scores_df.columns else "question"

    # Build lookup: question text → metric scores
    score_lookup = {}
    for _, row in scores_df.iterrows():
        q_text = row[df_q_col]
        score_lookup[q_text] = {
            name: (
                float(row[name])
                if name in row and not math.isnan(float(row[name]))
                else None
            )
            for name in METRIC_NAMES
            if name in row
        }

    # Patch scores into results
    for record in results:
        q_text = record.get("question", "")
        if q_text in score_lookup:
            for name in METRIC_NAMES:
                record[name] = score_lookup[q_text].get(name)
        else:
            for name in METRIC_NAMES:
                record[name] = None

    # Compute means
    mean_scores = {}
    for name in METRIC_NAMES:
        vals = [r[name] for r in results if r.get(name) is not None]
        mean_scores[name] = round(sum(vals) / len(vals), 4) if vals else None
        count  = len(vals)
        status = f"{mean_scores[name]:.4f}  ({count}/{len(results)} questions)" if mean_scores[name] else "N/A"
        print(f"  {name:<25}: {status}")

    return mean_scores


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_pipeline_results(pipeline_name: str, results: list[dict]):
    json_path = os.path.join(RESULTS_DIR, f"{pipeline_name}_results.json")
    csv_path  = os.path.join(RESULTS_DIR, f"{pipeline_name}_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"  Saved {pipeline_name} → {json_path}")


def save_ablation_summary(all_scores: dict):
    path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline", "Advanced", "Hybrid"])
        for metric_key in METRIC_NAMES:
            row = [metric_key]
            for pipeline in ["baseline", "advanced", "hybrid"]:
                val = all_scores[pipeline].get(metric_key)
                row.append(f"{val:.4f}" if val is not None else "N/A")
            writer.writerow(row)
    print(f"  Saved ablation summary → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(all_scores: dict):
    display = {
        "context_precision":  "Context Precision",
        "faithfulness":       "Faithfulness",
        "answer_relevancy":   "Answer Relevancy",
        "answer_correctness": "Answer Correctness",
    }
    print("\n" + "="*62)
    print(f"  {'Metric':<22} {'Baseline':>10} {'Advanced':>10} {'Hybrid':>10}")
    print("="*62)
    for key, label in display.items():
        row = f"  {label:<22}"
        for p in ["baseline", "advanced", "hybrid"]:
            val  = all_scores[p].get(key)
            cell = f"{val:.4f}" if val is not None else "N/A"
            row += f" {cell:>10}"
        print(row)
    print("="*62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start = datetime.now()
    ensure_results_dir()

    print("\n" + "█"*60)
    print("  CROSS-LINGUAL RAG — FULL EVALUATION (10-QUESTION SUBSET)")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("█"*60 + "\n")

    # Load golden dataset subset
    print("Loading golden dataset...")
    golden_data = load_golden_dataset()

    # Setup RAGAS judge (load once, reuse across all pipelines)
    ragas_llm, ragas_embeds = setup_ragas_judge()

    all_scores = {}

    # ── Run and evaluate each pipeline ───────────────────────────────────────
    for pipeline_name, run_fn in [
        ("baseline", run_baseline),
        ("advanced", run_advanced),
        ("hybrid",   run_hybrid),
    ]:
        # Step 1: Generate answers
        results = run_fn(golden_data)

        # Step 2: Evaluate with RAGAS
        mean_scores = evaluate_pipeline(
            pipeline_name, results, ragas_llm, ragas_embeds
        )
        all_scores[pipeline_name] = mean_scores

        # Step 3: Save results
        save_pipeline_results(pipeline_name, results)

    # ── Save and print final summary ──────────────────────────────────────────
    save_ablation_summary(all_scores)

    elapsed = (datetime.now() - start).seconds // 60
    print("\n" + "█"*60)
    print("  ABLATION STUDY RESULTS")
    print("█"*60)
    print_summary_table(all_scores)
    print(f"  Total time : ~{elapsed} minutes")
    print(f"  Results    : {RESULTS_DIR}")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()