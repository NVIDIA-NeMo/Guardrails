# GLiNER Configuration Examples

This directory contains configuration examples for using GLiNER in your NeMo Guardrails project.

For more details on the GLiNER integration, see [GLiNER Integration User Guide](../../../docs/user-guides/community/gliner.md).

## Structure

1. [pii_detection](./pii_detection) - Configuration for using GLiNER for PII detection.
2. [pii_masking](./pii_masking) - Configuration for using GLiNER for PII masking.

## Prerequisites

Start the GLiNER server before running these examples:

```bash
python nemoguardrails/library/gliner/gliner_server.py --host 0.0.0.0 --port 1235
```
