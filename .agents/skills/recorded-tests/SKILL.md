---
name: "recorded-tests"
description: "Decides when a code change requires a recorded/VCR test, whether a test belongs in tests/recorded/ or the plain unit suite under tests/, and how to structure it in either place. Use when adding, reviewing, or relocating tests that involve LLM calls, streaming, cassettes, snapshots, or provider traffic, and when changing provider clients, public API output shapes, or model-calling rails. Trigger keywords - recorded test, cassette, VCR, replay, snapshot test, tests/recorded, record-cassettes, fake cassette, TestChat, FakeLLMModel."
license: "Apache-2.0"
---

# Recorded Tests: When and How

`tests/recorded/` is a replay suite: it pins real provider wire traffic in
cassettes and pins the public API outputs produced from that traffic in inline
snapshots. Its value is refreshability: `make rewrite-cassettes` re-records
against live providers and detects drift. `make record-cassettes` runs record
mode `once`, so it only fills in cassettes that are missing and will not
refresh an existing one. A test that records no provider
traffic gets none of that value and pays all of the suite's overhead (config
directory, cassette plumbing, snapshot conventions, refresh workflow).

Read `tests/recorded/README.md` before writing any test in this suite. This
skill decides WHETHER and WHERE a recorded test is needed; the README defines
HOW the suite works.

## When a change requires a recorded test

The single criterion: a recorded test is required when the change alters bytes
sent to a provider, the interpretation of bytes received from one, or the
public output shape derived from them. Recorded tests are the only tests whose
expectations come from the provider rather than from us.

Concretely, add or update a recorded test when:

1. **The wire path changes**: `nemoguardrails/llm/clients/` (request
   construction, SSE parsing, error mapping, usage extraction). Fakes here
   only encode our own assumptions about the provider dialect. Division of
   labor: `tests/llm/clients/` with `httpx.MockTransport` owns synthetic wire
   conditions (status-code mapping, malformed bodies); `tests/recorded/clients/`
   owns real traffic (happy path, real error shapes).
2. **The public API output contract changes over provider-driven flows**:
   `generate_async` / `stream_async` / `check` output shapes, new
   `GenerationResponse` fields, usage metadata, streaming chunk framing. If
   such outputs change and no snapshot under `tests/recorded/rails/public_api/`
   moved, either coverage has a hole or the change did not do what you think.
3. **A new provider integration, engine, or model type is added**: minimum one
   happy-path recorded test plus one provider-error test (the invalid-model
   404 pattern), so the integration is drift-detectable from day one.
4. **A model-calling library rail is added or its prompt/parsing changes**
   (content safety, topic control, jailbreak detection, self check): the
   block/allow decision depends on real model output, so
   `tests/recorded/rails/library/` needs the outcome triad (allow, block,
   error). Pure Python rails such as `regex` need a suite entry too, since
   `nemoguardrails/library/README.md` requires recorded e2e coverage for
   every new library rail; they get non-vcr suite tests with their own config
   directory (exemplar: `tests/recorded/rails/library/test_regex.py`), not
   zero coverage. For the full contribution checklist of a new rail, see the
   `add-library-rail` skill.
5. **A bug fix whose trigger is provider-shaped data**: the bug only
   reproduces with a real payload (an SSE framing edge, a missing usage field,
   a provider error body variant). The regression test must be recorded, or a
   fake cassette per the README ladder if the payload is unrecordable.

A recorded test is NOT required when:

- The change is deterministic runtime logic, even when the diff touches
  `llmrails.py` (our own error envelopes, action dispatch, rail sequencing,
  fail-closed handling of local failures). Unit test in `tests/`.
- The change is a pure refactor. The recorded suite's job in a refactor is to
  be RUN, not extended: replay with `--block-network` before and after is the
  equivalence proof.
- The change is config or input validation, Colang runtime behavior, docs, or
  tooling.

File paths are a heuristic, not the rule: `llmrails.py` contains both
wire-adjacent contract code and deterministic logic in the same file. Always
apply the byte-level criterion above.

## Placement checklist

Answer these in order. The first "no" that routes you to `tests/` is final.

1. **Does the scenario send at least one real provider request?** Main model,
   a rail's model, or embeddings must actually cross the wire during the test.
   If nothing crosses the wire, there is nothing to record. Write a unit test
   in `tests/` using `TestChat` with `llm_completions` (see
   `tests/utils.py`), or `FakeLLM` directly.

   Red flags that mean the answer is "no":
   - The test passes `generator=` to `stream_async`, which bypasses the LLM
     entirely.
   - The only rail action involved is a local Python action registered by the
     test config (a fake checker, a deliberately failing action).
   - The assertion fires before any model call (input validation,
     `pytest.raises` on config errors).
   - You did not add `@pytest.mark.vcr` and the test still passes. No vcr
     mark means no cassette, which means it is a unit test wearing a
     recorded-suite costume.

2. **Is the interesting behavior in the provider response, or in our own
   deterministic logic?** Provider response shape (streaming chunk framing,
   error body format, usage metadata, tool-call structure, header handling)
   belongs here. Deterministic library logic belongs in `tests/`. The question
   to ask: "if the provider changed its behavior tomorrow, could this test's
   expected output change?" If it cannot, the test does not need a recording.

3. **Does an equivalent test already exist?** Search both `tests/` and
   `tests/recorded/` for the behavior. Never add a unit test and a recorded
   test for the same assertion in the same change; keep the unit test unless
   rule 2 says the behavior is provider-shaped.

4. **Is the scenario a failure that cannot occur against a live provider?**
   Follow the negative-path ladder in `tests/recorded/README.md`: prefer a
   recordable real error (for example an invalid model name yielding a real
   404), then pure runtime `pytest.raises`, and only as a last resort a
   `@pytest.mark.fake_cassette` with the required metadata. If the failure is
   injected by local test code rather than the provider, that is rule 1
   territory: unit test.

## Colang dialects

Recorded tests are dialect-single: do not duplicate a recorded test across
Colang 1 and Colang 2. For library rails the provider traffic is made by the
action, so both dialects record identical wire bytes; what varies between
dialects is deterministic flow routing, which unit tests own (both-dialect
`TestChat` coverage) together with the flow-files gate. Two exceptions:

- One v2 smoke pin in `tests/recorded/rails/library/` (a single rail blocked
  through the Colang 2 runtime) is worth its one cassette; a per-rail v2
  matrix is not.
- In `tests/recorded/rails/public_api/`, the v2 runtime is a different
  generation pipeline (different prompts and LLM calls for dialog and
  generation surfaces), so v2 coverage there records genuinely different
  provider traffic and is legitimate recorded-suite material.

## The narrow non-vcr exception

The suite intentionally hosts a small number of non-vcr, pure-runtime tests
(for example `StreamingNotSupportedError` validation, `check()` contracts
over `FakeLLM`) because `public_api` doubles as the public API surface
contract used for refactor equivalence proofs. This exception is a
convention, not an enforcement: nothing in `tests/recorded/conftest.py`
inspects markers, so a misplaced test collects and passes silently and no
tooling will tell you it is in the wrong suite. A non-vcr suite test carries
only the module-level `pytestmark = [pytest.mark.recorded,
pytest.mark.asyncio]` and no `vcr` mark (exemplar:
`tests/recorded/rails/library/test_regex.py`). Because nothing fails for you,
apply these criteria yourself. Add a non-vcr suite test only when all of
these hold:

- it pins a public API contract of the same surface as the neighboring
  recorded tests, or is input validation on that surface,
- it has no natural home in `tests/` (no equivalent unit test exists or could
  express the same public contract),
- it needs no new config directory of its own, EXCEPT for a rail under
  `tests/recorded/rails/library/`, where `nemoguardrails/library/README.md`
  mandates recorded coverage and a per-rail config directory is the required
  shape.

When in doubt, put it in `tests/`. Do not park a test here just because it
collects; nothing is checking, so the discipline is yours.

## Worked examples

Belongs in `tests/recorded/`:

- Streaming against a real OpenAI model, pinning chunk framing, final
  content, and usage metadata (`test_openai_stream_async_public_contract`).
  The chunk boundaries come from the provider's SSE stream; only a recording
  can pin them honestly.
- A content-safety rail whose checker model is a real provider call, pinning
  what the rail blocks. The block decision depends on the real model output.
- An invalid model name producing the provider's real 404, pinning that a
  failing safety model fails closed with `LLMCallException`. The error body
  is provider-shaped and refreshable.

Belongs in `tests/` instead:

- A rail action that raises `RuntimeError`, asserting the stream yields an
  internal-error envelope. The action is local, the envelope is constructed
  by our code, and `TestChat(llm_completions=...)` covers it with zero
  cassettes. (See `tests/test_streaming_internal_errors.py` for the canonical
  pattern.)
- Any assertion on our own JSON error format, rail ordering, or config
  validation.

## How, once placed in tests/recorded/

Follow `tests/recorded/README.md` exactly. The short version:

- Module-level `pytestmark = [pytest.mark.recorded, ...]`; add
  `@pytest.mark.vcr` per test when the module mixes vcr and non-vcr tests.
- Request credentials as fixture parameters (`openai_api_key`,
  `nvidia_api_key`); the fixture name is what gates recording.
- Config constants live in the suite-local `configs.py`; reuse an existing
  config directory before creating a new one.
- Record with `make record-cassettes RECORDED_TESTS=<node>
  RECORDED_REQUIRED_KEYS=<KEY>`, fill snapshots offline, then verify replay
  with `--block-network`. A new vcr test with no cassette committed is
  incomplete.
- Name negative tests `test_<surface>_<failure>_<behavior>` with `_raises`,
  `_fails_closed`, or `_invalid_*` suffixes.

## Review checklist for diffs touching tests/recorded/

- vcr-marked test with no committed cassette: incomplete, must be recorded.
- New non-vcr suite test: challenge it against the exception criteria above;
  it usually signals a test that belongs in `tests/`. Nothing enforces this,
  so read every test in the diff that has no `vcr` mark.
- New config directory for a single deterministic test: the test is in the
  wrong suite, unless it is the mandated per-rail config for a new
  `nemoguardrails/library/` rail.
- Same behavior asserted in both `tests/` and `tests/recorded/` in one
  change: keep one, prefer the unit test unless the behavior is
  provider-shaped.
- Diff changes `nemoguardrails/llm/clients/` or a model-calling library rail
  with no change under `tests/recorded/`: ask where the recorded coverage is.
