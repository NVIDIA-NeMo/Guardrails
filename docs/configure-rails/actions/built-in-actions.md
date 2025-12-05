---
title: Built-in Actions
description: Reference for default actions included in the NeMo Guardrails toolkit for common operations.
---

# Built-in Actions

This section describes the default actions included in the NeMo Guardrails toolkit.

## Core Actions

These actions are fundamental to the guardrails process:

| Action | Description |
|--------|-------------|
| `generate_user_intent` | Generate the canonical form for the user utterance |
| `generate_next_step` | Generate the next step in the conversation flow |
| `generate_bot_message` | Generate a bot message based on the desired intent |
| `retrieve_relevant_chunks` | Retrieve relevant chunks from the knowledge base |

### generate_user_intent

Converts raw user input into a canonical intent form:

```colang
# Automatically called during guardrails process
# Input: "Hello there!"
# Output: express greeting
```

### generate_next_step

Determines what the bot should do next:

```colang
# Automatically called to decide next action
# Output: bot express greeting, execute some_action, etc.
```

### generate_bot_message

Generates the actual bot response text:

```colang
# Converts intent to natural language
# Input: bot express greeting
# Output: "Hello! How can I help you today?"
```

### retrieve_relevant_chunks

Retrieves context from the knowledge base:

```colang
# Retrieves relevant documents for RAG
# Result stored in $relevant_chunks context variable
```

## Guardrail-Specific Actions

These actions implement built-in guardrails:

| Action | Description |
|--------|-------------|
| `self_check_input` | Check if user input should be allowed |
| `self_check_output` | Check if bot response should be allowed |
| `self_check_facts` | Verify factual accuracy of bot response |
| `self_check_hallucination` | Detect hallucinations in bot response |

### self_check_input

Validates user input against configured policies:

```yaml
# config.yml
rails:
  input:
    flows:
      - self check input
```

```colang
# rails/input.co
define flow self check input
  $allowed = execute self_check_input
  if not $allowed
    bot refuse to respond
    stop
```

### self_check_output

Validates bot output against configured policies:

```yaml
# config.yml
rails:
  output:
    flows:
      - self check output
```

```colang
# rails/output.co
define flow self check output
  $allowed = execute self_check_output
  if not $allowed
    bot refuse to respond
    stop
```

### self_check_facts

Verifies facts against retrieved knowledge base chunks:

```yaml
# config.yml
rails:
  output:
    flows:
      - self check facts
```

### self_check_hallucination

Detects hallucinated content in bot responses:

```yaml
# config.yml
rails:
  output:
    flows:
      - self check hallucination
```

## LangChain Tool Wrappers

The toolkit includes wrappers for popular LangChain tools:

| Action | Description | Requirements |
|--------|-------------|--------------|
| `apify` | Web scraping and automation | Apify API key |
| `bing_search` | Bing Web Search | Bing API key |
| `google_search` | Google Search | Google API key |
| `searx_search` | Searx search engine | Searx instance |
| `google_serper` | SerpApi Google Search | SerpApi key |
| `openweather_query` | Weather information | OpenWeatherMap API key |
| `serp_api_query` | SerpAPI search | SerpApi key |
| `wikipedia_query` | Wikipedia information | None |
| `wolfram_alpha_query` | Math and science queries | Wolfram Alpha API key |
| `zapier_nla_query` | Zapier automation | Zapier NLA API key |

### Using LangChain Tools

```colang
define flow answer with search
  user ask about current events
  $results = execute google_search(query=$user_query)
  bot provide search results
```

### Wikipedia Example

```colang
define flow answer with wikipedia
  user ask about historical facts
  $info = execute wikipedia_query(query=$user_query)
  bot provide information
```

## Sensitive Data Detection Actions

| Action | Description |
|--------|-------------|
| `detect_sensitive_data` | Detect PII in text |
| `mask_sensitive_data` | Mask detected PII |

### detect_sensitive_data

```yaml
# config.yml
rails:
  config:
    sensitive_data_detection:
      input:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - PHONE_NUMBER
```

```colang
define flow check input sensitive data
  $has_pii = execute detect_sensitive_data
  if $has_pii
    bot refuse to respond
    stop
```

### mask_sensitive_data

```colang
define flow mask input sensitive data
  $masked_input = execute mask_sensitive_data
  # Continue with masked input
```

## Content Safety Actions

| Action | Description |
|--------|-------------|
| `llama_guard_check_input` | LlamaGuard input moderation |
| `llama_guard_check_output` | LlamaGuard output moderation |
| `content_safety_check` | NVIDIA content safety model |

### LlamaGuard Example

```yaml
# config.yml
rails:
  input:
    flows:
      - llama guard check input
  output:
    flows:
      - llama guard check output
```

## Jailbreak Detection Actions

| Action | Description |
|--------|-------------|
| `check_jailbreak` | Detect jailbreak attempts |

```yaml
# config.yml
rails:
  input:
    flows:
      - check jailbreak
```

## Using Built-in Actions in Custom Flows

You can combine built-in actions with custom logic:

```colang
define flow enhanced_input_check
  # First, check for jailbreak
  $is_jailbreak = execute check_jailbreak
  if $is_jailbreak
    bot refuse to respond
    stop

  # Then, check for sensitive data
  $has_pii = execute detect_sensitive_data
  if $has_pii
    bot ask to remove sensitive data
    stop

  # Finally, run self-check
  $allowed = execute self_check_input
  if not $allowed
    bot refuse to respond
    stop
```

## Related Topics

- [Creating Custom Actions](creating-actions) - Create your own actions
- [Guardrails Library](../../user-guides/guardrails-library) - Complete guardrails reference
