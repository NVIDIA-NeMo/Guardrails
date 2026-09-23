# Context Bloat Detection

Detects context-manipulation attacks where attacker-controlled content (retrieved chunks or user
input) is padded, oversized, or repetitively structured to cause system prompt forgetting or
exhaust the token budget.

## Wiring

Add the flows you need to your `config.yml`:

```yaml
rails:
  retrieval:
    flows:
      - context bloat detection on retrieval

  input:
    flows:
      - context bloat detection on input
```

## Configuration

All fields are optional; defaults are shown below.

```yaml
rails:
  config:
    context_bloat_detection:
      # Size cap in characters. Inputs exceeding this are flagged.
      # Typically <5k for well-scoped agents.
      max_chars: 5000

      # Minimum characters before entropy/run/repetition checks apply.
      # Shorter texts are only checked against the size cap.
      min_chars: 50

      # Shannon entropy floor (bits per char).
      # English prose is roughly 4.0-4.5; padded/repetitive text drops below ~3.5.
      min_entropy: 3.5

      # Maximum fraction of repeated n-grams (0.0-1.0).
      # Values above 0.4 indicate padding-style repetition.
      max_repetition_ratio: 0.4

      # N-gram size used for repetition detection.
      ngram_size: 3

      # Maximum fraction of text that is the longest single-character run.
      # Catches "AAAAAAA..." or whitespace padding.
      max_run_ratio: 0.1

      # Action on detection:
      #   reject   -- abort the flow and return a user-facing message (recommended)
      #   truncate -- truncate to max_chars at the size cap, reject on all other checks
      #   warn     -- log only, do not modify or block
      action: reject
```
