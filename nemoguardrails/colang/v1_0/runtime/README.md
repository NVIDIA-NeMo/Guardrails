# Flows

The core abstraction in the NeMo Guardrails library is a **flow**.

## Events

The following is the list of standard events:

- `UtteranceUserActionFinished(final_transcript)`: a new utterance from the user has been received.
- `UserMessage(text)`: the final user message from the user has been decided (post input rails).
- `UserIntent(intent)`: a canonical form for the user utterance has been identified.
- `BotIntent(intent)`: a new bot intent has been decided i.e. what it should say.
- `StartUtteranceBotAction(content)`: the utterance for a bot message has been decided.
- `StartInternalSystemAction(action_name, is_system_action, action_parameters)`: it has been decided that an action should be started.
- `InternalSystemActionFinished(action_name, action_parameters, action_result)`: an action has finished.
- `Listen`: there's nothing let to do and the bot should listen for new user input/events.
- `ContextUpdate`: the context data has been updated.

## Context Variables

The following is the list of standard context variables:

- `last_user_message`: the last utterance from the user.
- `last_bot_message`: the last utterance from the bot.
- `relevant_chunks`: the relevant chunks of text related to what the user said.
