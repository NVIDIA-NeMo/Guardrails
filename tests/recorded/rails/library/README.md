# Recorded Library Rails

This suite records end-to-end behavior for flows in `nemoguardrails/library/`. Tests are grouped by behavior:

- `test_regex.py`
- `test_injection.py`
- `test_self_check.py`
- `test_content_safety.py`
- `test_topic_control.py`
- `test_jailbreak.py`
- `test_composition.py`

Add scenarios directly to the behavior module that owns them. Keep shared code limited to config constants in `configs.py` and execution helpers in `helpers.py`.

Use `check_async` for direct rail decisions. Use `generate_async` or `stream_async` only when that public API changes the behavior being asserted.

Provider-backed tests use pytest-recording default cassette names under:

```text
tests/recorded/rails/library/cassettes/<test_module>/<test_name>.yaml
```

When output rails need deterministic main-model text, prefer `FakeLLMModel` for generation or a supplied stream generator for streaming. Keep provider-backed rail calls on VCR cassettes when possible.

Run:

```bash
uv run pytest tests/recorded/rails/library --block-network -v
```

Refresh:

```bash
uv run pytest tests/recorded/rails/library --record-mode=rewrite -m "not fake_cassette" -v
```

## Jailbreak detection

`test_jailbreak.py` checks allow, block, and invalid-key behavior through the hosted
NVIDIA NemoGuard JailbreakDetect API. It verifies the provider's HTTP status and
classification as well as the public rail outcome, since this rail currently
allows requests on authorization errors. An HTTP 401 must not pass for a successful
benign classification. IORails parity reuses the same cassettes.

All three exchanges were recorded against the live hosted endpoint on 2026-11-04:
the benign `what can you do?` prompt, the DAN jailbreak prompt, and an HTTP 401
using `invalid-recorded-replay` as the deliberately invalid credential.
Credentials are stripped by the suite's recorder. The existing response sanitizer
normalizes jailbreak scores to `0.0` while preserving the classification, so these
tests pin the verdict rather than the exact confidence score.

Refresh the source recordings with a build.nvidia.com key authorized for this
endpoint:

```sh
make rewrite-cassettes \
  RECORDED_TESTS=tests/recorded/rails/library/test_jailbreak.py \
  RECORDED_REQUIRED_KEYS=NVIDIA_API_KEY
```

Hosted recordings do not execute the local Snowflake tokenizer. The separate
[`integration_tests/jailbreak_detection/`](../../../../integration_tests/jailbreak_detection/)
suite checks the overflow and cache regressions with the real, pinned tokenizer.
These recordings do not establish that the hosted service inspects every token of
an oversized input. The library's local 2,048-token rejection does not run for
NIM-backed requests; those depend on the endpoint's own input handling.
