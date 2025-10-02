# OpenAI Moderations API

NeMo Guardrails supports using the [OpenAI Moderations API](https://platform.openai.com/docs/guides/moderation) as an input or output rail out-of-the-box. You need to have the `OPENAI_API_KEY` environment variable set or configure it in your OpenAI client.

## Basic Usage

```yaml
rails:
  input:
    flows:
      # The simplified version using the flagged response
      - openai moderation

      # The detailed version with individual category checks
      # - openai moderation detailed
```

The `openai moderation` flow uses OpenAI's built-in flagging system to decide if the input should be allowed or not. The `openai moderation detailed` flow checks individual violation categories with custom logic.

## Using the Moderation API Endpoint

You can also use the moderation endpoint directly:

```yaml
rails:
  config:
    moderation:
      providers:
        - id: openai-moderation
          provider: openai
          model: omni-moderation-latest
          action: openai_moderation_api
          default: true
```

Then call the endpoint:
```bash
curl -X POST http://localhost:8000/v1/moderations \
  -H "Content-Type: application/json" \
  -d '{
    "input": "...text to classify goes here..."
  }'
```

## Supported Categories

OpenAI's moderation API detects the following categories:

| Category | Description |
|----------|-------------|
| `harassment` | Content that expresses, incites, or promotes harassing language towards any target |
| `harassment/threatening` | Harassment content that also includes violence or serious harm towards any target |
| `hate` | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste |
| `hate/threatening` | Hateful content that also includes violence or serious harm towards the targeted group |
| `illicit` | Content that gives advice or instruction on how to commit illicit acts |
| `illicit/violent` | Illicit content that also includes references to violence or procuring a weapon |
| `self-harm` | Content that promotes, encourages, or depicts acts of self-harm |
| `self-harm/intent` | Content where the speaker expresses intent to engage in acts of self-harm |
| `self-harm/instructions` | Content that encourages or gives instructions on acts of self-harm |
| `sexual` | Content meant to arouse sexual excitement or that promotes sexual services |
| `sexual/minors` | Sexual content that includes an individual who is under 18 years old |
| `violence` | Content that depicts death, violence, or physical injury |
| `violence/graphic` | Content that depicts death, violence, or physical injury in graphic detail |

## Customizing Thresholds

To customize the behavior, you can overwrite the [default flows](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/library/openai_moderate_text/flows.co) in your config. For example, to create a custom moderation flow:

```colang
define subflow custom openai moderation
  """Custom guardrail with specific threshold logic."""
  $result = execute openai_moderation_api

  # Block if OpenAI flags it as harmful
  if $result.get("flagged", False)
    bot refuse to respond
    stop

  # Custom threshold checks on category scores
  if $result.category_scores.get("violence", 0) > 0.5
    bot inform cannot engage in violent content
    stop
```

## Detailed Category Handling

Using OpenAI Text Moderation, you can control various violation categories individually. The API returns both boolean flags for each category and confidence scores. Here's an example of a detailed input moderation flow:

```colang
define flow openai input moderation detailed
  $result = execute openai_moderation_api(text=$user_message)

  if $result.categories.get("harassment", False)
    bot inform cannot engage in abusive or harmful behavior
    stop

  if $result.categories.get("hate", False)
    bot inform cannot engage in abusive or harmful behavior
    stop

  if $result.categories.get("sexual", False)
    bot inform cannot engage in inappropriate content
    stop

  if $result.categories.get("violence", False)
    bot inform cannot engage in abusive or harmful behavior
    stop

define bot inform cannot engage in abusive or harmful behavior
  "I will not engage in any abusive or harmful behavior."

define bot inform cannot engage in inappropriate content
  "I will not engage with inappropriate content."
```

## Using with Category Scores

You can also use the confidence scores for more nuanced control:

```colang
define subflow openai score based moderation
  """Moderation based on confidence scores rather than binary flags."""
  $result = execute openai_moderation_api

  # Custom thresholds for different categories
  if $result.category_scores.get("harassment", 0) > 0.7
    bot inform cannot engage in abusive or harmful behavior
    stop

  if $result.category_scores.get("hate", 0) > 0.6
    bot inform cannot engage in abusive or harmful behavior
    stop

  if $result.category_scores.get("violence", 0) > 0.8
    bot inform cannot engage in abusive or harmful behavior
    stop
```

## Environment Setup

Make sure you have your OpenAI API key configured:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or you can configure it in your application code when initializing the OpenAI client.

## Installation

The OpenAI moderation integration requires the `openai` package:

```bash
pip install openai
```

This is typically included when you install NeMo Guardrails with OpenAI support.
