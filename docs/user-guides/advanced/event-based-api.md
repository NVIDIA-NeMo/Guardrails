# Event-based API

You can use a guardrails configuration through an event-based API using [`LLMRails.generate_events_async`](../../api/nemoguardrails.rails.llm.llmrails.md#method-llmrailsgenerate_events_async) and [`LLMRails.generate_events](../../api/nemoguardrails.rails.llm.llmrails.md#method-llmrailsgenerate_events).

Example usage:

```python
import json
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("path/to/config")
app = LLMRails(config)

new_events = app.generate_events(events=[{
    "type": "UtteranceUserActionFinished",
    "final_transcript": "Hello! What can you do for me?"
}])
print(json.dumps(new_events, indent=True))
```

Example output:

```yaml
[
  {
    "type": "StartInternalSystemAction",
    "action_name": "generate_user_intent",
    "action_params": {},
    "action_result_key": null,
    "is_system_action": true,
  },
  {
    "type": "InternalSystemActionFinished",
    "action_name": "generate_user_intent",
    "action_params": {},
    "action_result_key": null,
    "status": "success",
    "return_value": null,
    "events": [{ "type": "UserIntent", "intent": "express greeting" }],
    "is_system_action": true,
  },
  { "type": "UserIntent", "intent": "express greeting" },
  { "type": "BotIntent", "intent": "express greeting" },
  {
    "type": "StartInternalSystemAction",
    "action_name": "retrieve_relevant_chunks",
    "action_params": {},
    "action_result_key": null,
    "is_system_action": true,
  },
  { "type": "ContextUpdate", "data": { "relevant_chunks": "" } },
  {
    "type": "InternalSystemActionFinished",
    "action_name": "retrieve_relevant_chunks",
    "action_params": {},
    "action_result_key": null,
    "status": "success",
    "return_value": "",
    "events": null,
    "is_system_action": true,
  },
  {
    "type": "StartInternalSystemAction",
    "action_name": "generate_bot_message",
    "action_params": {},
    "action_result_key": null,
    "is_system_action": true,
  },
  {
    "type": "ContextUpdate",
    "data": { "_last_bot_prompt": "<<REMOVED FOR READABILITY>>>" },
  },
  {
    "type": "InternalSystemActionFinished",
    "action_name": "generate_bot_message",
    "action_params": {},
    "action_result_key": null,
    "status": "success",
    "return_value": null,
    "events": [{ "type": "StartUtteranceBotAction", "script": "Hello!" }],
    "is_system_action": true,
  },
  { "type": "StartUtteranceBotAction", "script": "Hello!" },
  { "type": "Listen" },
]
```

## Event Types

NeMo Guardrails supports multiple types of events. Some are meant for internal use (e.g., `UserIntent`, `BotIntent`), while others represent the "public" interface (e.g., `UtteranceUserActionFinished`, `StartUtteranceBotAction`).

### `UtteranceUserActionFinished`

The raw message from the user.

Example:

```json
{
  "type": "UtteranceUserActionFinished",
  "final_transcript": "Hello!"
}
```

### `UserIntent`

The computed intent (a.k.a. canonical form) for what the user said.

Example:

```json
{
  "type": "UserIntent",
  "intent": "express greeting"
}
```

### `BotIntent`

The computed intent for what the bot should say.

Example:

```json
{
  "type": "BotIntent",
  "intent": "express greeting"
}
```

### `StartUtteranceBotAction`

The final message from the bot.

Example:

```json
{
  "type": "StartUtteranceBotAction",
  "script": "Hello!"
}
```

### `StartInternalSystemAction`

An action needs to be started.

Example:

```json
{
  "type": "StartInternalSystemAction",
  "action_name": "generate_user_intent",
  "action_params": {},
  "action_result_key": null,
  "is_system_action": true
}
```

### `InternalSystemActionFinished`

An action has finished.

Example:

```json
{
  "type": "InternalSystemActionFinished",
  "action_name": "generate_user_intent",
  "action_params": {},
  "action_result_key": null,
  "status": "success",
  "return_value": null,
  "events": [
    {
      "type": "UserIntent",
      "intent": "express greeting"
    }
  ],
  "is_system_action": true
}
```

### `ContextUpdate`

The context of the conversation has been updated.

Example:

```json
{
  "type": "ContextUpdate",
  "data": {
    "user_name": "John"
  }
}
```

### `listen`

The bot has finished processing the events and is waiting for new input.

Example:

```json
{
  "type": "Listen"
}
```

## Custom Events

You can also use custom events:

```json
{
  "type": "some_other_type",
  ...
}
```

**Note**: You need to make sure that the guardrails logic can handle the custom event. You do this by updating your flows to deal with the new events where needed. Otherwise, the custom event will just be ignored.

## Typical Usage

Typically, you will need to:

1. Persist the events history for a particular user in a database.
2. Whenever there is a new message or another event, you fetch the history and append the new event.
3. Use the guardrails API to generate the next events.
4. Filter the `StartUtteranceBotAction` events and return them to the user.
5. Persist the history of events back into the database.
