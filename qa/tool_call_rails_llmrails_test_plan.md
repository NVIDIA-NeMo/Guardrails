# QA Test Plan: LLMRails Tool-Call Rail Validation

## 1. Summary

This plan validates a legacy `LLMRails` fix for tool-call rails. When callers pass an assistant
message containing `tool_calls` and disable dialog rails with `options={"rails": {"dialog": False}}`,
the assistant tool-call message must still be converted into a `BotToolCalls` event so configured
`rails.tool_output` flows can inspect and block unsafe tool calls.

The change also updates generation-log processing so tool rail events appear in
`GenerationLog.activated_rails` as `tool_output` and `tool_input` rails.

## 2. Scope

| Dimension | In scope |
| --- | --- |
| Engine | Legacy `LLMRails` |
| Rails | `tool_output`, `tool_input` log visibility, existing text `output` rails |
| Interfaces | Python API / in-process validation |
| Provider | Model-free local config, no live provider |
| Modes | Non-streaming |

Out of scope: `IORails`, FastAPI server behavior, LangChain integration, streaming, live OpenAI or
other provider calls, performance/load testing.

## 3. Preconditions

- Checkout the branch containing the fix.
- Install project dependencies:

```bash
make install
```

- No live API keys are required.
- Run commands from the repository root.

## 4. Local Validation Script

Run:

```bash
uv run --locked python qa/validate_tool_call_rails.py
```

Expected output:

```text
TC-01: Tool output rails with dialog disabled
  Expected: Unsafe assistant tool call is blocked by the tool_output rail.
  Actual:   Assistant content: 'I cannot execute this tool request because the parameters may be unsafe.'
  Result:   PASS

TC-02: Plain text output rails regression path
  Expected: Plain assistant text is still evaluated by output rails when dialog is disabled.
  Actual:   Assistant content: 'The text output was blocked.'
  Result:   PASS

TC-03: Output rails and tool_output rails configured together
  Expected: Assistant tool calls are evaluated by tool_output rails, not blocked as empty text by output rails.
  Actual:   Assistant content: 'I cannot execute this tool request because the parameters may be unsafe.'
  Result:   PASS

TC-04: Safe tool call with list-form rails option
  Expected: Safe assistant tool call reaches tool_output rails and is not refused when options rails=['tool_output'].
  Actual:   Assistant content: '', validated_tools=['safe_tool']
  Result:   PASS

TC-05: Generation log tool rail entries
  Expected: Activated rails include tool_output/tool_input entries with names and durations.
  Actual:   types=['tool_output', 'tool_input'], names=['check tool call', 'check tool result'], durations=[0.25, 0.5]
  Result:   PASS

All tool-call rail validation checks passed.
```

The script is model-free and fails with an assertion error if any validation does not match the
expected behavior.

## 5. Simple End-To-End Flow

QA can validate the behavior with one basic Python flow and swap the Colang/YAML snippets below for
each scenario. The examples are model-free and do not require live provider credentials.

### 5.1 Basic End-To-End Python Example

```python
import asyncio

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action


@action(is_system_action=True)
async def validate_tool_parameters(tool_calls, context=None, **kwargs):
    tool_calls = tool_calls or (context.get("tool_calls", []) if context else [])

    for tool_call in tool_calls:
        args = tool_call.get("function", {}).get("arguments", {})
        for value in args.values():
            if isinstance(value, str) and "eval(" in value:
                return False

    return True


async def main():
    colang_content = """
    define subflow validate tool parameters
      $valid = execute validate_tool_parameters(tool_calls=$tool_calls)

      if not $valid
        bot refuse dangerous tool parameters
        abort

    define bot refuse dangerous tool parameters
      "I cannot execute this tool request because the parameters may be unsafe."
    """

    yaml_content = """
    models: []
    passthrough: true
    rails:
      tool_output:
        flows:
          - validate tool parameters
    """

    messages = [
        {"role": "user", "content": "Use the requested tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "dangerous_tool",
                        "arguments": {"param": "eval('malicious code')"},
                    },
                }
            ],
        },
    ]

    config = RailsConfig.from_content(colang_content, yaml_content)
    rails = LLMRails(config)
    rails.runtime.register_action(validate_tool_parameters, name="validate_tool_parameters")

    result = await rails.generate_async(
        messages=messages,
        options={"rails": {"dialog": False}},
    )

    print(result.response[0]["content"])


asyncio.run(main())
```

### 5.2 Tool Output Rail Only

Use this for TC-01 and TC-04.

```colang
define subflow validate tool parameters
  $valid = execute validate_tool_parameters(tool_calls=$tool_calls)

  if not $valid
    bot refuse dangerous tool parameters
    abort

define bot refuse dangerous tool parameters
  "I cannot execute this tool request because the parameters may be unsafe."
```

```yaml
models: []
passthrough: true
rails:
  tool_output:
    flows:
      - validate tool parameters
```

Example call:

```python
result = await rails.generate_async(
    messages=messages,
    options={"rails": {"dialog": False}},
)
```

### 5.3 Plain Text Output Rail Only

Use this for TC-02 to verify normal text-output rail behavior is unchanged.

```colang
define subflow block unsafe bot text
  if $bot_message == "unsafe text"
    bot refuse unsafe text
    abort

define bot refuse unsafe text
  "The text output was blocked."
```

```yaml
models: []
passthrough: true
rails:
  output:
    flows:
      - block unsafe bot text
```

Example call:

```python
result = await rails.generate_async(
    messages=[
        {"role": "user", "content": "Return the final answer"},
        {"role": "assistant", "content": "unsafe text"},
    ],
    options={"rails": {"dialog": False}},
)
```

### 5.4 Output And Tool Output Rails Together

Use this for TC-03 to verify an assistant tool-call message with empty text is routed to
`tool_output`, not mistakenly handled as an empty text response by normal `output` rails.

```colang
define subflow block empty bot text
  if $bot_message == ""
    bot refuse empty text
    abort

define bot refuse empty text
  "The empty text output was blocked."

define subflow validate tool parameters
  $valid = execute validate_tool_parameters(tool_calls=$tool_calls)

  if not $valid
    bot refuse dangerous tool parameters
    abort

define bot refuse dangerous tool parameters
  "I cannot execute this tool request because the parameters may be unsafe."
```

```yaml
models: []
passthrough: true
rails:
  output:
    flows:
      - block empty bot text
  tool_output:
    flows:
      - validate tool parameters
```

Example call:

```python
result = await rails.generate_async(
    messages=messages,
    options={"rails": {"dialog": False}},
)
```

## 6. Test Cases

### TC-01: Tool Output Rail Blocks Unsafe Assistant Tool Call With Dialog Disabled

| Field | Value |
| --- | --- |
| Rail | `tool_output` |
| Interface | Python `LLMRails.generate_async` |
| Priority | P0 |
| Type | Negative / regression |

Steps:

1. Configure `LLMRails` with a `tool_output` flow named `validate tool parameters`.
2. The validation action should reject tool call arguments containing unsafe content such as
   `eval(...)`.
3. Call `generate_async()` with messages ending in an assistant message containing `tool_calls`.
4. Pass `options={"rails": {"dialog": False}}`.

Expected:

- The assistant `tool_calls` message is preserved during preprocessing.
- The message is converted into a `BotToolCalls` event.
- The configured `tool_output` rail runs.
- The unsafe tool call is blocked and the response contains the configured refusal text.

Pass/fail:

- Pass if the unsafe tool call is blocked.
- Fail if the call is allowed through or the output rail does not run.

### TC-02: Plain Text Assistant Output Still Uses Existing Dialog-Disabled Path

| Field | Value |
| --- | --- |
| Rail | `output` |
| Interface | Python `LLMRails.generate_async` |
| Priority | P1 |
| Type | Positive / regression guard |

Steps:

1. Configure `LLMRails` with a normal text `output` rail that blocks a known assistant text value.
2. Call `generate_async()` with messages ending in a plain assistant message with no `tool_calls`.
3. Pass `options={"rails": {"dialog": False}}`.

Expected:

- Existing behavior is preserved for plain text assistant messages.
- The assistant text is still moved into `$bot_message`.
- The configured text output rail runs and blocks the unsafe text.
- No `BotToolCalls` event is expected for this case.

Pass/fail:

- Pass if the text output rail still blocks the plain assistant text.
- Fail if the behavior changes or the text output rail no longer runs.

### TC-03: Output Rails And Tool Output Rails Can Be Configured Together

| Field | Value |
| --- | --- |
| Rail | `output` and `tool_output` |
| Interface | Python `LLMRails.generate_async` |
| Priority | P0 |
| Type | Negative / routing regression |

Steps:

1. Configure `LLMRails` with both:
   - a normal `output` rail that blocks empty assistant text
   - a `tool_output` rail that blocks unsafe tool call arguments
2. Call `generate_async()` with messages ending in an assistant message that has empty `content`
   and unsafe `tool_calls`.
3. Pass `options={"rails": {"dialog": False}}`.

Expected:

- The assistant tool-call message is not treated as plain empty text.
- The normal `output` rail does not block the tool call as an empty bot message.
- The configured `tool_output` rail does run and blocks the unsafe tool call.
- The response contains the tool-output refusal text, not the plain text output refusal text.

Pass/fail:

- Pass if the tool-output rail blocks the unsafe tool call.
- Fail if the normal output rail handles the tool-call message as empty text, or if the tool call
  is allowed through.

### TC-04: Safe Tool Call Is Evaluated And Allowed With List-Form Rails Option

| Field | Value |
| --- | --- |
| Rail | `tool_output` |
| Interface | Python `LLMRails.generate_async` |
| Priority | P0 |
| Type | Positive / plugin-call-shape regression |

Steps:

1. Configure `LLMRails` with a `tool_output` flow named `validate tool parameters`.
2. The validation action should allow safe tool arguments such as `{"param": "safe value"}`.
3. Call `generate_async()` with messages ending in an assistant message containing a safe
   `tool_calls` payload.
4. Pass list-form generation options: `options={"rails": ["tool_output"]}`.

Expected:

- The list-form rails option normalizes correctly and enables `tool_output` while disabling other
  rail categories, including dialog rails.
- The assistant `tool_calls` message is preserved during preprocessing.
- The configured `tool_output` rail runs and sees the safe call.
- The safe call is not refused.

Pass/fail:

- Pass if the validation action receives the safe tool call and no refusal is produced.
- Fail if the safe call is blocked or does not reach the `tool_output` rail.

### TC-05: Generation Log Includes Tool Rail Entries

| Field | Value |
| --- | --- |
| Area | Generation logging |
| Interface | `compute_generation_log()` |
| Priority | P1 |
| Type | Observability regression |

Steps:

1. Produce or simulate a processing log containing:
   - `StartToolOutputRail` / `ToolOutputRailFinished`
   - `StartToolInputRail` / `ToolInputRailFinished`
2. Compute the `GenerationLog`.
3. Inspect `GenerationLog.activated_rails`.

Expected:

- `activated_rails` includes one `tool_output` entry and one `tool_input` entry.
- Rail names match the corresponding flow IDs.
- Durations are populated.
- Wrapper flows such as `process bot tool call`, `run tool output rails`, and
  `process user tool messages` are not misreported as dialog rails.

Pass/fail:

- Pass if tool rails are visible with the correct type, name, and duration.
- Fail if tool rails are missing, misclassified, or mixed with wrapper flows.

## 7. Automated Regression Tests

Run the focused regression tests:

```bash
make test TEST="tests/test_tool_output_rails.py::test_assistant_tool_calls_run_tool_output_rails_when_dialog_disabled tests/test_logging.py::test_compute_generation_log_includes_tool_rails"
```

Expected:

- Both tests pass.

## 8. Exit Criteria

- The local validation script passes.
- The two focused automated regression tests pass.
- QA confirms unsafe assistant tool calls are blocked when dialog rails are disabled.
- QA confirms plain text assistant output behavior is unchanged.
- QA confirms configs with both `output` and `tool_output` rails route assistant tool-call messages
  to `tool_output` rails.
- QA confirms safe assistant tool calls reach `tool_output` rails and are allowed with list-form
  `options={"rails": ["tool_output"]}`.
- QA confirms generation logs include `tool_output` and `tool_input` activated rails.

## 9. Notes And Risks

- This plan intentionally avoids live-provider testing. The bug is deterministic and occurs before
  any live model call is needed.
- This plan intentionally excludes `IORails`; that engine has a separate tool-calling rail path.
- If a downstream microservice validates tool-call history through another integration layer, run
  one additional smoke test through that service to confirm it passes assistant `tool_calls` into
  `LLMRails` with `dialog=False`.
