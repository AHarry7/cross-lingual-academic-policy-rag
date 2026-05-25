"""
eval/rerun_failed_metrics.py
─────────────────────────────
Re-evaluates only the metrics that produced NaN in the original run.
Loads answers + contexts from the already-saved JSON result files —
no pipeline re-run needed, no new Groq generation calls.

Fixes:
  - Baseline:  all metrics present — nothing to fix
  - Advanced:  context_precision = nan, answer_correctness = nan
  - Hybrid:    context_precision = nan

Runs RAGAS with concurrency=1 to avoid Groq rate-limit TimeoutErrors.

Usage (from project root):
  python -m eval.rerun_failed_metrics
"""

import os
import sys
import json
import csv
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# RAGAS
from ragas import evaluate
from ragas.metrics import context_precision, answer_correctness
from ragas.run_config import RunConfig
from datasets import Dataset

# LLM + Embeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
METRIC_NAMES = ["context_precision", "faithfulness", "answer_relevancy", "answer_correctness"]

# Only re-evaluate metrics that produced NaN
RERUN_PLAN = {
    "advanced": [context_precision, answer_correctness],
    "hybrid":   [context_precision],
}
RERUN_NAMES = {
    "advanced": ["context_precision", "answer_correctness"],
    "hybrid":   ["context_precision"],
}


def setup_judge():
    print("Loading judge LLM and embeddings...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    ragas_llm = LangchainLLMWrapper(llm)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    print("  Judge ready.\n")
    return ragas_llm, ragas_embeddings


def load_pipeline_results(pipeline_name):
    path = os.path.join(RESULTS_DIR, f"{pipeline_name}_results.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pipeline_results(pipeline_name, results):
    json_path = os.path.join(RESULTS_DIR, f"{pipeline_name}_results.json")
    csv_path  = os.path.join(RESULTS_DIR, f"{pipeline_name}_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Updated JSON -> {json_path}")

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    print(f"  Updated CSV  -> {csv_path}")


def rerun_metrics(pipeline_name, results, metrics, metric_names, ragas_llm, ragas_embeddings):
    print(f"\nRe-evaluating [{pipeline_name.upper()}]: {metric_names}")

    # Auto-detect actual key names — RAGAS may rename columns when saving
    sample = results[0]
    print(f"  Detected JSON keys: {list(sample.keys())}")

    q_key  = next((k for k in sample if k in ("question", "user_input")), None)
    a_key  = next((k for k in sample if k in ("answer", "response")), None)
    c_key  = next((k for k in sample if k in ("contexts", "retrieved_contexts")), None)
    gt_key = next((k for k in sample if k in ("ground_truth", "reference")), None)

    if not all([q_key, a_key, c_key, gt_key]):
        raise ValueError(
            f"Could not map required keys.\n"
            f"Available: {list(sample.keys())}\n"
            f"Detected: q={q_key}, a={a_key}, c={c_key}, gt={gt_key}"
        )
    print(f"  Mapped: q={q_key}, a={a_key}, c={c_key}, gt={gt_key}")

    dataset = Dataset.from_dict({
        "question":     [r[q_key]  for r in results],
        "answer":       [r[a_key]  for r in results],
        "contexts":     [r[c_key]  for r in results],
        "ground_truth": [r[gt_key] for r in results],
    })

    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings") and ragas_embeddings:
            metric.embeddings = ragas_embeddings

    # KEY FIX: max_workers=1 means one LLM call at a time — no rate limit bursts
    run_config = RunConfig(
        max_workers=1,
        max_retries=3,
        timeout=120,
    )

    eval_result = evaluate(dataset, metrics=metrics, run_config=run_config)
    scores_df   = eval_result.to_pandas()

    # Patch per-question scores back into results list
    for i, row in scores_df.iterrows():
        for name in metric_names:
            if name in row:
                val = float(row[name])
                results[i][name] = val if not math.isnan(val) else None

    # Compute means from patched results
    mean_scores = {}
    for name in metric_names:
        vals = [r[name] for r in results if r.get(name) is not None]
        mean_scores[name] = round(sum(vals) / len(vals), 4) if vals else None
        print(f"  {name}: {mean_scores[name]}")

    return mean_scores


def load_existing_summary():
    path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    summary = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["Metric"]
            for pipeline in ["Baseline", "Advanced", "Hybrid"]:
                if pipeline not in summary:
                    summary[pipeline] = {}
                val = row.get(pipeline, "N/A")
                try:
                    summary[pipeline][metric] = float(val)
                except (ValueError, TypeError):
                    summary[pipeline][metric] = None
    return summary


def save_corrected_summary(all_scores):
    path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + list(all_scores.keys()))
        for metric_key in METRIC_NAMES:
            row = [metric_key]
            for pipeline in all_scores:
                val = all_scores[pipeline].get(metric_key)
                row.append(f"{val:.4f}" if val is not None else "N/A")
            writer.writerow(row)
    print(f"\n  Saved corrected summary -> {path}")


def print_summary_table(all_scores):
    pipelines = list(all_scores.keys())
    col_w = 12
    display = {
        "context_precision":  "Context Precision",
        "faithfulness":       "Faithfulness",
        "answer_relevancy":   "Answer Relevancy",
        "answer_correctness": "Answer Correctness",
    }

    print("\n")
    print("=" * 64)
    print(f"  {'Metric':<22} {'Baseline':>10} {'Advanced':>10} {'Hybrid':>10}")
    print("=" * 64)
    for key, label in display.items():
        row = f"  {label:<22}"
        for p in pipelines:
            val = all_scores[p].get(key)
            cell = f"{val:.4f}" if val is not None else "N/A"
            row += f" {cell:>10}"
        print(row)
    print("=" * 64)
    print()


def main():
    print("\n" + "=" * 60)
    print("  RE-RUNNING FAILED METRICS (concurrency=1)")
    print("=" * 60 + "\n")

    ragas_llm, ragas_embeds = setup_judge()

    # Start from the existing valid scores
    all_scores = load_existing_summary()

    for pipeline_name, metrics in RERUN_PLAN.items():
        metric_names = RERUN_NAMES[pipeline_name]
        results      = load_pipeline_results(pipeline_name)

        new_scores = rerun_metrics(
            pipeline_name, results, metrics, metric_names,
            ragas_llm, ragas_embeds,
        )

        # Patch corrected scores into the summary
        cap = pipeline_name.capitalize()
        for name, score in new_scores.items():
            all_scores[cap][name] = score

        save_pipeline_results(pipeline_name, results)

    save_corrected_summary(all_scores)

    print("\n" + "=" * 60)
    print("  CORRECTED ABLATION STUDY RESULTS")
    print("=" * 60)
    print_summary_table(all_scores)


if __name__ == "__main__":
    main()