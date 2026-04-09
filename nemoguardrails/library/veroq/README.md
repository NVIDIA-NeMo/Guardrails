# VeroQ Shield

This integration uses [VeroQ Shield](https://veroq.ai) to fact-check LLM output in real time.

VeroQ Shield extracts verifiable claims from the bot response, checks each claim against live sources, and returns a trust score with per-claim verdicts. Responses containing contradicted claims are blocked.

## Setup

Install the `aiohttp` package (included with NeMo Guardrails) and get a VeroQ API key:

1. Sign up at [veroq.ai](https://veroq.ai) to get an API key.
2. Set the `VEROQ_API_KEY` environment variable:

```bash
export VEROQ_API_KEY="your-api-key"
```

## Usage

Add the `veroq check output facts` flow as an output rail in your `config.yml`:

**Colang 2.x:**

```yaml
rails:
  output:
    flows:
      - veroq check output facts
```

**Colang 1.x:**

```yaml
rails:
  output:
    flows:
      - veroq check output facts
```

The guardrail will block bot responses where:
- Any extracted claim is contradicted by verified sources, or
- The overall trust score is below 0.7.

## How It Works

1. The bot generates a response.
2. VeroQ extracts up to 5 verifiable claims from the response.
3. Each claim is fact-checked against live sources.
4. If any claim is contradicted or the trust score is low, the response is blocked and replaced with a safe fallback message.

## Learn More

- [VeroQ Documentation](https://docs.veroq.ai)
- [VeroQ Shield](https://veroq.ai/shield)
