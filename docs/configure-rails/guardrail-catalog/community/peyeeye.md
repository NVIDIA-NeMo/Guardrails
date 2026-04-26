# Peyeeye Integration

[Peyeeye](https://peyeeye.ai) is a PII redaction & rehydration API. The
integration lets NeMo Guardrails redact PII from a user's message before it is
sent to the LLM, and then rehydrate the model's response so the user sees the
original values — all without the LLM ever seeing the raw PII.

## How it works

1. **Input rail**: the user message is sent to `POST /v1/redact`, which returns
   the message with each detected entity swapped for a stable placeholder
   (e.g. `[EMAIL_1]`) plus a session id.
2. **LLM call**: the redacted message goes to the LLM. The LLM's response
   typically echoes some placeholders (`"I have emailed you at [EMAIL_1]"`).
3. **Output rail**: the response is sent to `POST /v1/rehydrate` with the
   stored session id; the placeholders are swapped back to the original
   values before the response is returned to the user.

Two session modes:

- **`stateful`** (default): peyeeye stores the token→value mapping under a
  `ses_…` id. The integration sends a best-effort `DELETE /v1/sessions/{id}`
  after rehydration.
- **`stateless`**: peyeeye returns a sealed AEAD blob (`skey_…`) and retains
  nothing server-side. Rehydration is performed by passing that blob back.

## Setup

1. Create an account at [peyeeye.ai](https://peyeeye.ai) and grab an API key.
2. Set the `PEYEEYE_API_KEY` environment variable.
3. Update your `config.yml`:

```yaml
rails:
  config:
    peyeeye:
      # Optional; defaults to https://api.peyeeye.ai. Override for self-hosted.
      api_base: https://api.peyeeye.ai
      input:
        # Optional list of entity IDs. Omit to use the default catalog.
        entities:
          - EMAIL
          - PHONE
          - CARD
        locale: auto
        session_mode: stateful
      output:
        # Output is rehydrated via the input session, so options here only
        # apply if you also enable `peyeeye redact retrieval` etc.
        locale: auto
  input:
    flows:
      - peyeeye redact input
  output:
    flows:
      - peyeeye rehydrate output
```

The `peyeeye redact input` flow stashes the session id in
`$peyeeye_input_session_id`; `peyeeye rehydrate output` reads it.

## Retrieval flow

If you want to redact retrieved knowledge-base chunks before they are
concatenated into the LLM prompt, add the retrieval flow:

```yaml
rails:
  retrieval:
    flows:
      - peyeeye redact retrieval
```

## Configuration reference

| Field | Default | Notes |
| --- | --- | --- |
| `api_base` | `https://api.peyeeye.ai` | Override via `PEYEEYE_API_BASE`. |
| `input.entities` | `null` | List of entity IDs to restrict detection to. |
| `input.locale` | `"auto"` | BCP-47 or `auto`. |
| `input.session_mode` | `"stateful"` | `"stateful"` or `"stateless"`. |
| `output.*` / `retrieval.*` | same as input | Per-source overrides. |

## Error handling

The integration intentionally **does not** silently forward unredacted text:

- A 401 response raises `PEyeEyeGuardrailMissingSecrets`.
- Any other 4xx/5xx, malformed JSON, or response with a different shape than
  expected raises `PEyeEyeGuardrailAPIError`.
- A length mismatch between the request and response (`/v1/redact` returning
  fewer texts than were sent) raises `PEyeEyeGuardrailAPIError` instead of
  guessing.

Rehydration is forgiving — failures are logged and the original (still
redacted) text is returned to the caller, so a transient outage on the
rehydrate endpoint does not leak placeholders or break the response.

## Notes

- The action handlers are in `nemoguardrails/library/peyeeye/actions.py`.
- The Colang flow definitions live in
  `nemoguardrails/library/peyeeye/flows.co` (Colang 2) and
  `flows.v1.co` (Colang 1).
- For more on the API surface and supported entities, see the
  [peyeeye documentation](https://peyeeye.ai).
