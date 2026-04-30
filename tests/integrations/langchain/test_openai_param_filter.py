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
  baseline JSON of probe results and asserts is_openai_reasoning_model has zero
  false negatives against it. Runs in normal CI. Catches accidental
  classifier regressions.

- test_probe_openai_live: gated on LIVE_TEST_MODE / TEST_LIVE_MODE and
  OPENAI_API_KEY. Hits /v1/models and /v1/chat/completions to verify the
  classifier still matches current OpenAI API behavior. Run this quarterly
  or after bumping langchain-openai.

Each candidate model is probed concurrently against every parameter listed
in PROBES. A model is treated as a reasoning model (in the classifier
sense) if the API rejects any of those params. The baseline records the
per-param verdict ("accepted" / "rejected" / "other: ...") so future
diffs surface OpenAI behavior changes per param.

Regenerate the baseline file (after confirming classifier changes):

    UPDATE_BASELINE=1 OPENAI_API_KEY=sk-... \\
        poetry run python tests/integrations/langchain/test_openai_param_filter.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from nemoguardrails.llm.openai_reasoning import is_openai_reasoning_model

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


PROBES: dict[str, dict] = {
    "stop": {"stop": ["User:"]},
    "temperature": {"temperature": 0.5},
    # max_tokens probe: do not also send max_completion_tokens (the default
    # body field) so a "use max_completion_tokens instead" rejection on
    # reasoning models is unambiguous.
    "max_tokens": {"max_tokens": 1},
}


def _load_baseline() -> list[dict]:
    payload = json.loads(BASELINE_PATH.read_text())
    return payload["results"]


def _evaluate(probe_results: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (false_negatives, ignored_false_positives).

    A model is considered to behave as a reasoning model (in the
    classifier's sense) if it rejects any probed param. PROBES is
    deliberately limited to params nemoguardrails actually sends from
    internal action code (temperature, stop, max_tokens); other OpenAI
    params are out of scope here.
    """
    false_negatives = []
    false_positives = []
    for result in probe_results:
        predicted = is_openai_reasoning_model(result["model"])
        rejects = any(result.get(param) == "rejected" for param in PROBES)
        if rejects and not predicted:
            false_negatives.append(result)
        elif predicted and not rejects:
            false_positives.append(result["model"])
    return false_negatives, false_positives


def test_classifier_matches_baseline():
    results = _load_baseline()
    assert results, "baseline is empty, regenerate via UPDATE_BASELINE=1"
    missing_fields = [result["model"] for result in results if any(param not in result for param in PROBES)]
    assert not missing_fields, (
        "Baseline is missing per-param probe results; regenerate via "
        f"UPDATE_BASELINE=1. Missing entries for: {sorted(set(missing_fields))[:5]}..."
    )
    false_negatives, false_positives = _evaluate(results)
    probed_list = ", ".join(sorted(PROBES))
    assert not false_negatives, (
        f"Classifier says these models do NOT reject any of {{{probed_list}}}, "
        "but the committed probe baseline shows they DO. Update "
        f"is_openai_reasoning_model or regenerate baseline: {false_negatives}"
    )
    unexpected = set(false_positives) - KNOWN_FALSE_POSITIVES
    assert not unexpected, (
        f"Classifier strips reasoning-only params for models that accept all of {{{probed_list}}}: "
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
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    }
    # Only set the default token field when the probe itself is not testing
    # max_tokens; otherwise the body would carry both fields and OpenAI's
    # response signal becomes ambiguous.
    if "max_tokens" not in extra:
        body["max_completion_tokens"] = 1
    body.update(extra)
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
    message = err.get("message", "") or ""
    if err.get("param") == param or f"'{param}'" in message:
        return "rejected"
    # Bare-keyword case: rejection mentions the param without single quotes
    # and ``err.param`` is None, e.g. 403 "You are not allowed to request
    # logprobs from this model" or 400 "specify max_completion_tokens rather
    # than max_tokens". We require both the keyword and a rejection-shaped
    # phrase so requirement-shaped messages ("max_tokens parameter is
    # required") fall through to accepted-by-inference instead.
    msg_lower = message.lower()
    rejection_indicators = (
        "not allowed",
        "not supported",
        "rather than",
        "instead",
        "unsupported",
        "deprecated",
    )
    if re.search(rf"\b{re.escape(param)}\b", msg_lower) and any(p in msg_lower for p in rejection_indicators):
        return "rejected"
    # OpenAI validates params before generation, so a max_completion_tokens /
    # max_tokens / output-limit error on a 1-token probe means the param was
    # accepted and generation started. Treat as accepted-by-inference.
    if "max_completion_tokens" in message or "max_tokens" in message or "output limit" in message:
        return "accepted"
    # Preserve the API's error type + message so baseline diffs surface why a
    # model is neither accepted nor rejected (eg "only supported in v1/responses").
    err_type = err.get("type", "error")
    return f"other: {err_type}: {message}" if message else f"http_{resp.status_code}"


async def _probe_model(client: httpx.AsyncClient, api_key: str, model: str) -> dict:
    # Independent probes per param run concurrently; models are still
    # iterated sequentially by the caller to avoid hitting OpenAI rate limits.
    names = list(PROBES.keys())
    responses = await asyncio.gather(*(_post(client, api_key, model, PROBES[name]) for name in names))
    result: dict = {"model": model}
    for name, resp in zip(names, responses):
        result[name] = _classify(resp, name)
    return result


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
        f"Live OpenAI probe found models the classifier misses. Update is_openai_reasoning_model: {false_negatives}"
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
