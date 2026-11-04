# Bundled jailbreak input tokenizer

`tokenizer.json.gz` is a lossless gzip copy of the unmodified `tokenizer.json` from
[`Snowflake/snowflake-arctic-embed-m-long`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long)
at revision `92d97331f1f4b6a366c1f161354b9f3390cc219f`.

- Source: https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long/resolve/92d97331f1f4b6a366c1f161354b9f3390cc219f/tokenizer.json
- SHA-256 of the decompressed source: `91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854`
- License: Apache-2.0, as declared by the upstream model card; see `LICENSE.txt`.

The library uses this artifact to enforce the detector's 2,048-token coverage
before cache lookup and remote HTTP calls. It needs no model weights, PyTorch,
Transformers, Hugging Face credentials, or network access. The loader verifies
the checksum before use.

To reproduce the compressed artifact after downloading the exact source above:

```python
import gzip
from pathlib import Path

source = Path("tokenizer.json").read_bytes()
Path("tokenizer.json.gz").write_bytes(gzip.compress(source, mtime=0))
```
