# Glossary

Below are the main concepts used in NeMo Guardrails:

- **LLM-based Application**: a software application that uses an LLM to drive
- **Bot**: synonym for *LLM-based application*.
- **Utterance**: the raw text coming from the user or the bot.
- **Intent**: the canonical form (i.e. structured representation) of a user/bot utterance.
- **Event**: something that has happened and is relevant to the conversation e.g. user is silent, user clicked something, user made a gesture, etc.
- **Action**: a custom code that the bot can invoke; usually for connecting to third-party API.
- **Context**: any data relevant to the conversation (i.e. a key-value dictionary).
- **Flow**: a sequence of messages and events, potentially with additional branching logic.
- **Rails**: specific ways of controlling the behavior of a conversational system (a.k.a. bot) e.g. not talk about politics, respond in a specific way to certain user requests, follow a predefined dialog path, use a specific language style, extract data etc.

## Recommended naming conventions

User messages:
- the first word should be a verb; "ask", "respond", "inform", "provide", "express", "comment", "confirm", "deny", "request"
- the rest of the words should be nouns
- should read naturally (e.g. not `user credit card problem` vs. `user inform credit card problem`)
