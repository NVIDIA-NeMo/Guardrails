# Regular Expression Detection Integration

The NVIDIA NeMo Guardrails library provides out-of-the-box support for content moderation based on regular expression (regex) pattern matching. This integration enables you to detect and block content that matches specific patterns in user inputs, bot outputs, and retrieved knowledge base chunks.

## Overview

Regex detection is useful for scenarios such as:

- Blocking specific keywords or phrases
- Detecting patterns like Social Security Numbers, credit card numbers, or other sensitive data formats
- Filtering profanity or inappropriate content using basic regex pattern matching

## Setup

No additional packages are required. The regex detection rails is built into the NVIDIA NeMo Guardrails library.

## Usage

The regex detection rails uses a flexible configuration that allows you to define specific regex patterns for input, output, and retrieval checking.

### Configuration Structure

Add the regex configuration to your `config.yml` file:

```yaml
rails:
  config:
    regex_detection:
      input:
        patterns:
          - "\\b(password|secret|api[_-]?key)\\b"
          - "\\d{3}-\\d{2}-\\d{4}"  # SSN pattern
        case_insensitive: true
      output:
        patterns:
          - "\\bconfidential\\b"
          - "\\binternal[_-]?use[_-]?only\\b"
        case_insensitive: true
      retrieval:
        patterns:
          - "\\bclassified\\b"
        case_insensitive: false
  input:
    flows:
      - regex check input
  output:
    flows:
      - regex check output
  retrieval:
    flows:
      - regex check retrieval
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `patterns` | List[str] | `[]` | List of regex patterns to match against the text |
| `case_insensitive` | bool | `false` | Whether to perform case-insensitive matching |

### Input Rails

To detect regex patterns in user input:

```yaml
rails:
  config:
    regex_detection:
      input:
        patterns:
          - "\\b(hack|exploit|bypass)\\b"
        case_insensitive: true
  input:
    flows:
      - regex check input
```

### Output Rails

To detect regex patterns in bot output:

```yaml
rails:
  config:
    regex_detection:
      output:
        patterns:
          - "\\bdo not share\\b"
        case_insensitive: true
  output:
    flows:
      - regex check output
```

### Retrieval Rails

When a chunk matches any of the configured retrieval patterns, **that chunk is removed** from the retrieval results and is not passed to the model. Only chunks that do not match the patterns are kept and used as context.

To detect regex patterns in retrieved knowledge base chunks:

```yaml
rails:
  config:
    regex_detection:
      retrieval:
        patterns:
          - "\\bproprietary\\b"
        case_insensitive: true
  retrieval:
    flows:
      - regex check retrieval
```

## Complete Example

Here's a comprehensive example configuration:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4

rails:
  config:
    regex_detection:
      input:
        patterns:
          # Block sensitive data patterns
          - "\\d{3}-\\d{2}-\\d{4}"          # SSN
          - "\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}"  # Credit card
          - "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"  # Email
          # Block specific keywords
          - "\\b(password|secret|api[_-]?key|token)\\b"
        case_insensitive: true
      output:
        patterns:
          - "\\b(confidential|internal[_-]?only|do[_-]?not[_-]?share)\\b"
        case_insensitive: true

  input:
    flows:
      - regex check input

  output:
    flows:
      - regex check output
```

## Return Value

The `detect_regex_pattern` action returns a `RegexDetectionResult` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `is_match` | bool | `True` if any pattern matched, `False` otherwise. |
| `text` | str | The original text that was checked. |
| `detections` | List[str] | List of the pattern strings that matched. Empty if no match. |

For example, if the input text `"my SSN is 123-45-6789"` is checked against a pattern `\\d{3}-\\d{2}-\\d{4}`, the result would be:

```json
{
  "is_match": true,
  "text": "my SSN is 123-45-6789",
  "detections": ["\\d{3}-\\d{2}-\\d{4}"]
}
```

## Regex Pattern Tips

When writing Python regex patterns, keep in mind:

1. **Escape backslashes**: In YAML, use double backslashes (`\\`) for regex escape sequences (e.g., `\\b` for word boundary, `\\d` for digit).

2. **Word boundaries**: Use `\\b` to match whole words only. For example, `\\bpassword\\b` matches "password" but not "passwords" or "mypassword".

3. **Character classes**: Use `[_-]?` to match optional separators (e.g., `api[_-]?key` matches "apikey", "api-key", and "api_key").

4. **Common patterns**:
   - SSN: `\\d{3}-\\d{2}-\\d{4}`
   - Credit card: `\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}`
   - Email: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}`
   - Phone: `\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}`

For more information about Python regex rules, please refer to the [official documentation](https://docs.python.org/3/howto/regex.html#regex-howto).

## Regex Rail Behavior

When a regex rail has a regex pattern match, the following steps happen (depending on rails type):

**Input and output rails**

1. The action returns a `RegexDetectionResult` with `is_match=True` and the matched pattern(s) in `detections`.
2. The bot responds with: `"I'm sorry, I can't respond to that."` (using the standard `bot refuse to respond` message).
3. The flow is aborted.

**Retrieval rails**

1. Chunks that match any retrieval pattern are **removed** from the retrieval results. Only non-matching chunks are passed to the model as context.

**NOTE:** The matched pattern(s) are available in `result["detections"]` for logging and debugging. They are also logged at `INFO` level by the action.
