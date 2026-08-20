"""
evaluation/run_evaluation.py
─────────────────────────────
Ragas evaluation script.
Reads the golden dataset and runs faithfulness metric.
Exits with code 1 if faithfulness < 0.90.

RAG Inference: Cerebras gpt-oss-120b — OpenAI-compatible endpoint
Ragas Judge:   Cerebras gpt-oss-120b — 120B model for reliable JSON-structured evaluation
  Both avoid Groq's per-minute token-bucket rate limits.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.append(str(BACKEND_DIR))

from app.config import get_settings
from app.rag.chain import execute_rag_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

# ─── Rate Limit Config ────────────────────────────────────────────────────────
# Groq free tier: ~30 RPM per model. Sleep 6s between questions to keep
# combined usage (RAG inference + Ragas evaluation) under the limit.
SLEEP_BETWEEN_CALLS_SECS = 6.0

FAITHFULNESS_THRESHOLD = 0.90


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def run_single_question(question: str) -> Dict:
    """Run one question through the RAG pipeline with retry on 429."""
    result = await execute_rag_pipeline(question=question)
    return result


async def collect_results(questions: List[Dict]) -> List[Dict]:
    """Iterate through the golden dataset, respecting rate limits."""
    results = []
    total = len(questions)

    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info("[%d/%d] Running: %s", i + 1, total, question[:80])

        try:
            result = await run_single_question(question)
            results.append({
                "question": question,
                "answer": result.get("answer") or "",
                # A refusal is a *correct* outcome when the context lacks the answer,
                # so it is tracked separately instead of masquerading as an empty answer.
                "refusal": result.get("refusal") or "",
                # Use the FULL page_content, not the 200-char content_preview.
                # Ragas faithfulness requires the complete retrieved text to verify claims.
                "contexts": result.get("full_contexts") or [
                    chunk.content_preview
                    for chunk in result.get("retrieved_chunks", [])
                ],
                "ground_truth": ground_truth,
                "error": "",
            })
        except Exception as e:
            logger.error("Failed on question %d: %s", i + 1, e)
            results.append({
                "question": question,
                "answer": "",
                "refusal": "",
                "contexts": [],
                "ground_truth": ground_truth,
                "error": f"{type(e).__name__}: {e}",
            })

        # Enforce rate limit — sleep between questions
        if i < total - 1:
            logger.debug("Sleeping %.1fs to respect RPM limit...", SLEEP_BETWEEN_CALLS_SECS)
            await asyncio.sleep(SLEEP_BETWEEN_CALLS_SECS)

    return results


def run_ragas_evaluation(results: List[Dict]) -> Dict[str, float]:
    """Run Ragas metrics on collected results."""
    settings = get_settings()

    # Using Cerebras via OpenAI-compatible client — gpt-oss-120b handles Ragas JSON prompts reliably
    ragas_llm = ChatOpenAI(
        model="gpt-oss-120b",
        temperature=0.0,
        max_tokens=4096,
        api_key=settings.cerebras_api_key,
        base_url="https://api.cerebras.ai/v1",
    )
    # Ragas only needs these four columns; drop our bookkeeping fields.
    dataset = Dataset.from_list([
        {k: r[k] for k in ("question", "answer", "contexts", "ground_truth")}
        for r in results
    ])

    logger.info("Running Ragas evaluation metrics (Max workers=1 to prevent 429s)...")
    run_config = RunConfig(max_workers=1, max_retries=15, max_wait=60)
    score = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
        run_config=run_config,
    )

    return score


async def main():
    logger.info("Loading golden dataset from %s", GOLDEN_DATASET_PATH)

    with open(GOLDEN_DATASET_PATH, "r") as f:
        questions = json.load(f)

    import os
    # Reuse cached inference only when it came from a clean run of the current
    # schema. Rows that errored out (not rows that legitimately refused) mean the
    # cache is stale and inference has to run again.
    results = None
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            cached = json.load(f)
        stale_schema = any("refusal" not in r for r in cached)
        errored = sum(1 for r in cached if r.get("error"))
        if stale_schema or errored:
            reason = "written by an older version" if stale_schema else f"contains {errored} failed rows"
            logger.warning("Cache at %s %s — re-running inference.", RESULTS_PATH, reason)
            os.remove(RESULTS_PATH)
        else:
            logger.info("Found valid cache at %s (%d entries), skipping inference.", RESULTS_PATH, len(cached))
            results = cached

    if results is None:
        logger.info("Running %d questions through the RAG pipeline...", len(questions))
        results = await collect_results(questions)

        # Save raw results
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Raw results saved to %s", RESULTS_PATH)

    # Partition the run. A refusal is correct behaviour when the retrieved context
    # cannot support an answer, so it is reported as its own rate rather than being
    # scored for faithfulness (a refusal is never "grounded in the context") or
    # silently dropped, which would flatter the gate.
    errored = [r for r in results if r.get("error")]
    refused = [r for r in results if not r.get("error") and not r["answer"].strip()]
    answered = [r for r in results if not r.get("error") and r["answer"].strip()]

    logger.info(
        "Run breakdown: %d answered, %d refused, %d errored (of %d)",
        len(answered), len(refused), len(errored), len(results),
    )
    for r in errored:
        logger.error("  pipeline error on %r -> %s", r["question"][:70], r["error"])
    for r in refused:
        logger.info("  refused: %r", r["question"][:70])

    if not answered:
        logger.error("GATE FAILED: no question produced an answer to score.")
        return False

    scores = run_ragas_evaluation(answered)

    # Safely extract aggregate scores from EvaluationResult
    import math
    def _nan_safe_mean(values):
        valid = [v for v in values if v is not None and not math.isnan(v)]
        return sum(valid) / len(valid) if valid else 0.0

    if hasattr(scores, "items"):
        scores_dict = {k: v for k, v in scores.items()}
    elif isinstance(scores, dict):
        scores_dict = scores
    else:
        try:
            scores_dict = dict(scores)
        except Exception:
            scores_dict = {}

    # Ragas EvaluationResult.scores is a list of per-row dicts — compute NaN-safe mean
    if not scores_dict or all(v == 0.0 for v in scores_dict.values()):
        s = getattr(scores, "scores", [])
        if isinstance(s, list) and len(s) > 0:
            logger.info("Computing NaN-safe mean from %d per-row score dicts...", len(s))
            keys = s[0].keys()
            scores_dict = {k: _nan_safe_mean([row.get(k) for row in s]) for k in keys}

    faithfulness_score = scores_dict.get("faithfulness", 0.0)
    if math.isnan(faithfulness_score):
        faithfulness_score = 0.0
        
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS (%d of %d questions scored)", len(answered), len(results))
    # Only faithfulness is computed above — printing metrics that never ran as
    # 0.000 made passing runs look like failures.
    for name, value in scores_dict.items():
        logger.info("  %-18s %.3f", name.replace("_", " ").title() + ":", value)
    logger.info("  %-18s %.3f (%d/%d)", "Refusal rate:",
                len(refused) / len(results), len(refused), len(results))
    logger.info("=" * 60)

    # ── CI/CD Gate ────────────────────────────────────────────────────────────
    if errored:
        logger.error("GATE FAILED: %d question(s) crashed the pipeline.", len(errored))
        return False

    if faithfulness_score < FAITHFULNESS_THRESHOLD:
        logger.error(
            "GATE FAILED: Faithfulness score %.3f is below threshold %.2f",
            faithfulness_score,
            FAITHFULNESS_THRESHOLD,
        )
        return False
    else:
        logger.info(
            "GATE PASSED: Faithfulness %.3f >= %.2f",
            faithfulness_score,
            FAITHFULNESS_THRESHOLD,
        )
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
