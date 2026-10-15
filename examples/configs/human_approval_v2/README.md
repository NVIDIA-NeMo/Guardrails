# Human approval (Colang v2 output rail)

Example configuration for the `human_approval` library rail. Bot output that matches
configured regex patterns requires a human reviewer to reply with an approval keyword
before the original text is released.

## Usage

```colang
import core
import llm
import guardrails
import nemoguardrails.library.human_approval

flow main
  activate llm continuation

flow output rails $output_text
  await human approval on bot output $output_text
```

## Multi-turn behavior ([#2067](https://github.com/NVIDIA-NeMo/Guardrails/issues/2067))

This library flow **does not use `abort` on rejection** — it sets `$bot_message` to the
rejection message and completes normally so `$output_rails_in_progress` clears in
`guardrails.co`. Approval prompts therefore fire again on later turns in the same session.

Custom output rails that still call `abort` may hit [#2067](https://github.com/NVIDIA-NeMo/Guardrails/issues/2067)
(output rails skipped on subsequent turns until upstream fixes flag cleanup).
