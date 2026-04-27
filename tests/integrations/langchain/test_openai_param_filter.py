# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI reasoning-model param-filter validation.

Two layers of coverage:

- test_classifier_matches_baseline: fast, no-network. Reads the committed
  baseline JSON of probe results and asserts _is_openai_reasoning_model has zero
  false negatives against it. Runs in normal CI. Catches accidental
  classifier regressions.

- test_probe_openai_live: gated on LIVE_TEST_MODE / TEST_LIVE_MODE and
  OPENAI_API_KEY. Hits /v1/models and /v1/chat/completions to verify the
  classifier still matches current OpenAI API behavior. Run this quarterly
  or after bumping langchain-openai.

Regenerate the baseline file (after confirming classifier changes):

    UPDATE_BASELINE=1 OPENAI_API_KEY=sk-... \\
        poetry run python tests/integrations/langchain/test_openai_param_filter.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from nemoguardrails.integrations.langchain.llm_adapter import _is_openai_reasoning_model

BASELINE_PATH = Path(__file__).parent / "data" / "openai_reasoning_probe_baseline.json"

LIVE_TEST_MODE = os.environ.get("LIVE_TEST_MODE") or os.environ.get("TEST_LIVE_MODE")

CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
MODELS_URL = "https://api.openai.com/v1/models"

NON_CHAT_HINTS = (
    "audio",
    "realtime",
    "tts",
    "image",
    "transcribe",
    "moderation",
    "embedding",
    "search",
)

# Models our classifier flags as reasoning but which are not usable via
# /v1/chat/completions at all (they route to /v1/responses or /v1/completions,
# or are codex-style non-chat models). Over-stripping params on them is
# irrelevant because any actual call would fail upstream. Kept explicit so
# that unexpected new false positives (ones that would affect real users)
# fail the test instead of silently piling up.
KNOWN_FALSE_POSITIVES = frozenset(
    {
        "gpt-5-codex",
        "gpt-5-pro",
        "gpt-5-pro-2025-10-06",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5.2-codex",
        "gpt-5.2-pro",
        "gpt-5.2-pro-2025-12-11",
        "gpt-5.3-codex",
        "gpt-5.4-pro",
        "gpt-5.4-pro-2026-03-05",
        "gpt-5.5-pro",
        "gpt-5.5-pro-2026-04-23",
        "o1-pro",
        "o1-pro-2025-03-19",
        "o3-pro",
        "o3-pro-2025-06-10",
    }
)


def _load_baseline() -> list[dict]:
    payload = json.loads(BASELINE_PATH.read_text())
    return payload["results"]


def _evaluate(probe_results: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (false_negatives, ignored_false_positives)."""
    false_negatives = []
    false_positives = []
    for result in probe_results:
        predicted = _is_openai_reasoning_model(result["model"])
        rejects = result["stop"] == "rejected" or result["temperature"] == "rejected"
        if rejects and not predicted:
            false_negatives.append(result)
        elif predicted and not rejects:
            false_positives.append(result["model"])
    return false_negatives, false_positives


def test_classifier_matches_baseline():
    results = _load_baseline()
    assert results, "baseline is empty, regenerate via UPDATE_BASELINE=1"
    false_negatives, false_positives = _evaluate(results)
    assert not false_negatives, (
        "Classifier says these models do NOT reject stop/temperature, "
        "but the committed probe baseline shows they DO. Update "
        f"_is_openai_reasoning_model or regenerate baseline: {false_negatives}"
    )
    unexpected = set(false_positives) - KNOWN_FALSE_POSITIVES
    assert not unexpected, (
        "Classifier strips stop/temperature for models that actually accept them: "
        f"{sorted(unexpected)}. If intentional, add them to KNOWN_FALSE_POSITIVES."
    )


def _is_chat_candidate(model_id: str) -> bool:
    # Add new family prefixes here as OpenAI releases them (o5, gpt-6, ...).
    # Probing without a known prefix would pull in embeddings/audio/moderation
    # models that aren't chat-completions at all.
    mid = model_id.lower()
    if not any(mid.startswith(p) for p in ("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return False
    return not any(k in mid for k in NON_CHAT_HINTS)


async def _fetch_models(client: httpx.AsyncClient, api_key: str) -> list[str]:
    r = await client.get(MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    r.raise_for_status()
    ids = [m["id"] for m in r.json()["data"]]
    return sorted({m for m in ids if _is_chat_candidate(m)})


async def _post(client: httpx.AsyncClient, api_key: str, model: str, extra: dict) -> httpx.Response:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 1,
        **extra,
    }
    return await client.post(
        CHAT_COMPLETIONS_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )


def _classify(resp: httpx.Response, param: str) -> str:
    if resp.status_code == 200:
        return "accepted"
    try:
        err = resp.json().get("error", {})
    except Exception:
        return f"http_{resp.status_code}"
    message = err.get("message", "")
    if err.get("param") == param or f"'{param}'" in message:
        return "rejected"
    # OpenAI validates params before generation, so a max_tokens/output-limit
    # error on a 1-token probe means the param was accepted and generation
    # started. Treat as accepted-by-inference.
    if "max_tokens" in message or "output limit" in message:
        return "accepted"
    # Preserve the API's error type + message so baseline diffs surface why a
    # model is neither accepted nor rejected (eg "only supported in v1/responses").
    err_type = err.get("type", "error")
    return f"other: {err_type}: {message}" if message else f"http_{resp.status_code}"


async def _probe_model(client: httpx.AsyncClient, api_key: str, model: str) -> dict:
    # Two independent probes per model run concurrently; models are still
    # iterated sequentially by the caller to avoid hitting OpenAI rate limits
    # with ~130 concurrent requests on a large account.
    stop_resp, temp_resp = await asyncio.gather(
        _post(client, api_key, model, {"stop": ["User:"]}),
        _post(client, api_key, model, {"temperature": 0.5}),
    )
    return {
        "model": model,
        "stop": _classify(stop_resp, "stop"),
        "temperature": _classify(temp_resp, "temperature"),
    }


async def _probe_all(client: httpx.AsyncClient, api_key: str) -> list[dict]:
    models = await _fetch_models(client, api_key)
    return [await _probe_model(client, api_key, model) for model in models]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not LIVE_TEST_MODE,
    reason="LIVE_TEST_MODE or TEST_LIVE_MODE must be set to run the live OpenAI probe",
)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY must be set",
)
async def test_probe_openai_live():
    async with httpx.AsyncClient() as client:
        results = await _probe_all(client, os.environ["OPENAI_API_KEY"])

    assert results, "probe returned no models"
    false_negatives, _ = _evaluate(results)
    assert not false_negatives, (
        f"Live OpenAI probe found models the classifier misses. Update _is_openai_reasoning_model: {false_negatives}"
    )


async def _regenerate_baseline() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required to regenerate baseline", file=sys.stderr)
        return 2
    async with httpx.AsyncClient() as client:
        print("Fetching model list and probing...")
        results = await _probe_all(client, api_key)
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "captured_at": datetime.now(timezone.utc).date().isoformat(),
        "openai_chat_completions_url": CHAT_COMPLETIONS_URL,
        "results": sorted(results, key=lambda r: r["model"]),
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(results)} models to {BASELINE_PATH}")
    false_negatives, false_positives = _evaluate(results)
    print(f"FN={len(false_negatives)}  FP (harmless)={len(false_positives)}")
    if false_negatives:
        for r in false_negatives:
            print(f"  FN: {r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_regenerate_baseline()))
