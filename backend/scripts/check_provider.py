"""
scripts/check_provider.py
─────────────────────────
Answers one question before you deploy: can the configured LLM provider
actually serve this pipeline?

A key that authenticates is not enough — this project needs the model to honour
`with_structured_output(..., method="json_mode")`, because citations, refusals,
and confidence all ride on that schema. A provider can accept your key, bill you
nothing, and still be unusable here because it ignores json_mode.

    python scripts/check_provider.py                 # check what .env selects
    python scripts/check_provider.py --provider gemini --model gemini-2.0-flash
    python scripts/check_provider.py --all-models    # probe every model offered

Exit code is 0 only if the provider passes every check.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from app.config import LLM_ENDPOINTS, PROVIDER_KEY_FIELDS, get_settings  # noqa: E402

# One chunk of context and a question it can answer, plus a question it cannot.
# The refusal case matters: a model that answers it from world knowledge will
# hallucinate in production no matter how good its json_mode support is.
CONTEXT = (
    "--- Chunk ID: chunk_test01 | Source: system_architecture.md ---\n"
    "This RAG system uses the all-MiniLM-L6-v2 sentence-transformer model from "
    "HuggingFace for generating 384-dimensional dense vector embeddings."
)
GROUNDED_Q = "What embedding model does this system use?"
UNANSWERABLE_Q = "What is the population of Jakarta?"


def _client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def check_model(base_url: str, api_key: str, model: str) -> bool:
    from langchain_openai import ChatOpenAI

    from app.models import RAGResponse

    llm = ChatOpenAI(model=model, temperature=0.0, api_key=api_key, base_url=base_url)

    # 1. Plain completion — is the key live and the model id correct?
    try:
        llm.invoke("Reply with exactly: OK")
    except Exception as e:
        code = getattr(e, "status_code", None) or getattr(e, "code", "?")
        print(f"    ✗ completion failed (HTTP {code}): {str(e)[:150]}")
        return False
    print("    ✓ completion")

    structured = llm.with_structured_output(RAGResponse, method="json_mode")
    system = (
        "Answer ONLY from the context. Cite the CHUNK_IDs supporting your answer. "
        "If the context cannot answer, leave 'answer' empty and write a full "
        "sentence in 'refusal'. Reply as JSON with keys: answer, citations, "
        "refusal, confidence.\n\n"
    )

    # 2. Structured output on an answerable question.
    try:
        out = structured.invoke(f"{system}Context:\n{CONTEXT}\n\nQuestion: {GROUNDED_Q}")
    except Exception as e:
        print(f"    ✗ json_mode failed: {type(e).__name__}: {str(e)[:150]}")
        return False

    if "MiniLM" not in (out.answer or ""):
        print(f"    ✗ answer not grounded in context: {(out.answer or '')[:90]!r}")
        return False
    if "chunk_test01" not in (out.citations or []):
        print(f"    ✗ citation missing or wrong: {out.citations}")
        return False
    print(f"    ✓ json_mode + citation  (confidence {out.confidence})")

    # 3. Refusal behaviour — the anti-hallucination guarantee.
    try:
        ref = structured.invoke(f"{system}Context:\n{CONTEXT}\n\nQuestion: {UNANSWERABLE_Q}")
    except Exception as e:
        print(f"    ✗ refusal probe failed: {type(e).__name__}: {str(e)[:120]}")
        return False

    if not ref.refusal or (ref.answer or "").strip():
        print(f"    ✗ did NOT refuse an unanswerable question "
              f"(answer={(ref.answer or '')[:60]!r})")
        return False
    print(f"    ✓ refuses cleanly: {ref.refusal[:70]!r}")
    return True


def list_models(base_url: str, api_key: str) -> List[str]:
    try:
        return sorted(m.id for m in _client(base_url, api_key).models.list().data)
    except Exception as e:
        print(f"  (could not list models: {type(e).__name__}: {str(e)[:110]})")
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", help=f"one of {sorted(LLM_ENDPOINTS)}")
    ap.add_argument("--model", help="model id to test")
    ap.add_argument("--all-models", action="store_true",
                    help="probe every model the provider lists")
    args = ap.parse_args()

    settings = get_settings()
    provider = (args.provider or settings.llm_provider).lower()
    if provider not in LLM_ENDPOINTS:
        print(f"unknown provider {provider!r}; expected one of {sorted(LLM_ENDPOINTS)}")
        return 2

    base_url = LLM_ENDPOINTS[provider]
    field = PROVIDER_KEY_FIELDS[provider]
    api_key = getattr(settings, field, "") or os.environ.get(field.upper(), "")
    if not api_key:
        print(f"No key for {provider!r}. Set {field.upper()} in backend/.env")
        return 2

    print(f"provider : {provider}")
    print(f"endpoint : {base_url}")
    print(f"key      : {api_key[:6]}…{api_key[-4:]} (len {len(api_key)})\n")

    models: List[Optional[str]]
    if args.all_models:
        models = list_models(base_url, api_key)
        skip = ("whisper", "tts", "embed", "guard", "orpheus", "rerank")
        models = [m for m in models if not any(s in m.lower() for s in skip)]
        print(f"probing {len(models)} model(s)\n")
    else:
        models = [args.model or settings.llm_model]

    passed = []
    for m in models:
        print(f"  {m}")
        if check_model(base_url, api_key, m):
            passed.append(m)
        print()

    if not passed:
        print("RESULT: no working model — this provider cannot serve the pipeline.")
        return 1

    print(f"RESULT: {len(passed)} model(s) usable: {', '.join(passed)}")
    print("\nTo use it, set in backend/.env (and in your HF Space secrets):")
    print(f"  LLM_PROVIDER={provider}")
    print(f"  LLM_MODEL={passed[0]}")
    print(f"  {field.upper()}=<your key>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
