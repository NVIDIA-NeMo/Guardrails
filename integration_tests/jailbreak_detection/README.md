# Jailbreak detector tokenizer regressions

These tests run the production `SnowflakeEmbed.__call__` with the real Snowflake
`AutoTokenizer` and CPU PyTorch tensors. Transformer and random-forest inference
are mocked; no model weights or provider credentials are needed.

They check the 2,048-token boundary, including special tokens, and a suffix beyond
that boundary. A second regression uses actual WordPiece tokenization to show why
whitespace-normalized cache keys can bypass the length check: the BERT normalizer
removes U+001C between `a` and `b`, producing one token, while `a b` produces two.
The cache normalizer treats those strings as equivalent.

Use a separate environment for the optional packages, then run from the repository
root. This keeps the default development environment unchanged:

```sh
JAILBREAK_TEST_ENV=$(mktemp -d /tmp/nemo-jailbreak-tokenizer.XXXXXX)
UV_PROJECT_ENVIRONMENT="$JAILBREAK_TEST_ENV" uv sync --locked
uv pip install --python "$JAILBREAK_TEST_ENV/bin/python" \
  'transformers==5.3.0' 'torch==2.14.1' \
  'tokenizers==0.22.2' 'huggingface-hub==1.16.1'
UV_PROJECT_ENVIRONMENT="$JAILBREAK_TEST_ENV" UV_NO_SYNC=1 \
  uv run --locked pytest integration_tests/jailbreak_detection -q
```

The first run downloads only four tokenizer files from the public
[`Snowflake/snowflake-arctic-embed-m-long`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long)
repository, under Apache-2.0, at revision
`92d97331f1f4b6a366c1f161354b9f3390cc219f`. The test verifies the original
`tokenizer.json` SHA-256:
`91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854`.
It disables remote code and authentication for the download.

After warming the Hugging Face cache, run without network access:

```sh
HF_HUB_OFFLINE=1 UV_PROJECT_ENVIRONMENT="$JAILBREAK_TEST_ENV" UV_NO_SYNC=1 \
  uv run --locked pytest integration_tests/jailbreak_detection -q
```

Missing `transformers` or `torch` skips the suite. These tests sit outside the
default unit-test paths so unit tests never download artifacts.
