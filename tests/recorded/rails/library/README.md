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
poetry run pytest tests/recorded/rails/library --block-network -v
```

Refresh:

```bash
poetry run pytest tests/recorded/rails/library --record-mode=rewrite -m "not fake_cassette" -v
```
