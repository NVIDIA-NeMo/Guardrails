# Retrieval-Augmented Generation

This guide shows how to apply a guardrails configuration in a RAG scenario. This guide builds on the [previous guide](../6-topical-rails/README.md), developing further the demo ABC Bot.

## Prerequisites

1. Install the `openai` package:

```bash
pip install openai
```

2. Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY=$OPENAI_API_KEY    # Replace with your own key
```

3. If you're running this inside a notebook, patch the AsyncIO loop.

```python
import nest_asyncio

nest_asyncio.apply()
```

## Usage

There are two modes in which you can use a guardrails configuration in conjunction with RAG:

1. **Relevant Chunks**: perform the retrieval yourself and pass the **relevant chunks** directly to the `generate` method.
2. **Knowledge Base**: configure a **knowledge base** directly into the guardrails configuration and let NeMo Guardrails manage the retrieval part.

### Relevant Chunks

In the previous guide, the message "How many free vacation days do I have per year" yields a general response:

```python
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate(messages=[{
    "role": "user",
    "content": "How many vacation days do I have per year?"
}])
print(response["content"])
```

```
Full-time employees are eligible for up to two weeks of paid vacation time per year. Part-time employees receive a prorated amount based on their hours worked. Please refer to the employee handbook for more information.
```

ABC company's Employee Handbook contains the following information:

```markdown
Employees are eligible for the following time off:

* Vacation: 20 days per year, accrued monthly.
* Sick leave: 15 days per year, accrued monthly.
* Personal days: 5 days per year, accrued monthly.
* Paid holidays: New Year's Day, Memorial Day, Independence Day, Thanksgiving Day, Christmas Day.
* Bereavement leave: 3 days paid leave for immediate family members, 1 day for non-immediate family members.
```

You can pass this information directly to guardrails when making a `generate` call:

```python
response = rails.generate(messages=[{
    "role": "context",
    "content": {
        "relevant_chunks": """
            Employees are eligible for the following time off:
              * Vacation: 20 days per year, accrued monthly.
              * Sick leave: 15 days per year, accrued monthly.
              * Personal days: 5 days per year, accrued monthly.
              * Paid holidays: New Year's Day, Memorial Day, Independence Day, Thanksgiving Day, Christmas Day.
              * Bereavement leave: 3 days paid leave for immediate family members, 1 day for non-immediate family members. """
    }
},{
    "role": "user",
    "content": "How many vacation days do I have per year?"
}])
print(response["content"])
```

```
Eligible employees receive 20 days of paid vacation time per year, which accrues monthly. You can find more information about this in the employee handbook.
```

As expected, the response contains the correct answer.

### Knowledge Base

There are three ways you can configure a knowledge base directly into a guardrails configuration:

1. Using the *kb* folder.
2. Using a custom `retrieve_relevant_chunks` action.
3. Using a custom `EmbeddingSearchProvider`.

For option 1, you can add a knowledge base directly into your guardrails configuration by creating a *kb* folder inside the *config* folder and adding documents there. Currently, only the Markdown format is supported. For a quick example, check out the complete implementation of the [ABC Bot](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/examples/bots/abc/README.md).

Options 2 and 3 represent advanced use cases beyond the scope of this topic.

## Wrapping Up

This guide introduced how a guardrails configuration can be used in the context of a RAG setup.

## Next

To continue learning about NeMo Guardrails, check out:
1. [Guardrails Library](../../../docs/user-guides/guardrails-library.md).
2. [Configuration Guide](../../../docs/user-guides/configuration-guide.md).
