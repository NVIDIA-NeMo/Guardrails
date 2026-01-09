# PolicyAI Integration

NeMo Guardrails supports using the [PolicyAI](https://musubilabs.ai) content moderation API as an input and output rail out-of-the-box (you need to have the `POLICYAI_API_KEY` environment variable set).

PolicyAI provides flexible policy-based content moderation, allowing you to define custom policies for your specific use cases and manage them through tags.

## Setup

1. Sign up for a PolicyAI account at [musubilabs.ai](https://musubilabs.ai)
2. Create your policies and organize them with tags
3. Set the required environment variables:

```bash
export POLICYAI_API_KEY="your-api-key"
export POLICYAI_BASE_URL="https://api.musubilabs.ai"  # Optional, this is the default
export POLICYAI_TAG_NAME="prod"  # Optional, defaults to "prod"
```

## Usage

### Basic Input Moderation

```yaml
rails:
  input:
    flows:
      - policyai moderation on input
```

### Basic Output Moderation

```yaml
rails:
  output:
    flows:
      - policyai moderation on output
```

### Using Different Tags

To use different policy tags for different environments, set the `POLICYAI_TAG_NAME` environment variable:

```bash
# For staging environment
export POLICYAI_TAG_NAME="staging"

# For production environment
export POLICYAI_TAG_NAME="prod"
```

## Complete Example

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4

rails:
  input:
    flows:
      - policyai moderation on input

  output:
    flows:
      - policyai moderation on output
```

## How It Works

1. **Input Rails**: When a user sends a message, PolicyAI evaluates it against all policies attached to the configured tag. If any policy returns `UNSAFE`, the message is blocked.

2. **Output Rails**: Before the bot's response is sent to the user, PolicyAI evaluates it. If the content violates any policy, the response is replaced with a refusal message.

## Response Format

PolicyAI returns the following information for each evaluation:

- `assessment`: `"SAFE"` or `"UNSAFE"`
- `category`: The category of violation (if UNSAFE)
- `severity`: Severity level from 0 (safe) to 3 (high severity)
- `reason`: Human-readable explanation

## Customizing Behavior

To customize the behavior when content is flagged, you can override the default flows in your config:

```text
define subflow policyai moderation on input
  """Custom PolicyAI input moderation."""
  $result = execute call_policyai_api(text=$user_message)

  if $result.assessment == "UNSAFE"
    bot inform content policy violation
    stop

define bot inform content policy violation
  "I'm sorry, but I cannot process that request. Please rephrase your message."
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POLICYAI_API_KEY` | Yes | - | Your PolicyAI API key |
| `POLICYAI_BASE_URL` | No | `https://api.musubilabs.ai` | PolicyAI API base URL |
| `POLICYAI_TAG_NAME` | No | `prod` | Default policy tag to use |

## Error Handling

If the PolicyAI API is unavailable or returns an error, the action will raise an exception. To implement fail-open or fail-closed behavior, you can wrap the action in a try-catch block in your custom flows.

## Learn More

- [PolicyAI Documentation](https://docs.musubilabs.ai)
- [Musubi Labs](https://musubilabs.ai)
