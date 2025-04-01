# Cleanlab Integration

Cleanlab's state-of-the-art [LLM uncertainty estimator](https://cleanlab.ai/blog/trustworthy-language-model/) scores the trustworthiness of any LLM response, to detect incorrect/hallucinated outputs in real-time.

In question-answering / RAG applications: high trustworthiness is indicative of correct responses, while in general open-ended applications, a high score corresponds to the response being helpful and informative. Low trustworthiness scores are typically incorrect/bad outputs, or complex prompts where the LLM might have output the right response this time but may output the wrong response when run on the same prompt again (so it cannot be trusted).

The trustworthiness score is further explained and comprehensively benchmarked in [Cleanlab's documentation](https://help.cleanlab.ai/tlm/).

The `cleanlab trustworthiness` guardrail flow uses a default trustworthiness score threshold of 0.6 to determine if your LLM output should be allowed or not (i.e., if the trustworthiness score is below the threshold, the response is flagged as  "untrustworthy"). You can easily change the cutoff value for the trustworthiness score by adjusting the threshold in the [config](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/nemoguardrails/library/cleanlab/flows.co). For example, to change the threshold to 0.7, you can add the following flow to your config:

```colang
define subflow cleanlab trustworthiness
  """Guardrail based on trustworthiness score."""
  $result = execute call cleanlab api

  if $result.trustworthiness_score < 0.7
    bot response untrustworthy
    stop

define bot response untrustworthy
  "Don't place much confidence in this response"
```

## Setup

Install the Python client to use Cleanlab's trustworthiness score:

```
pip install cleanlab-studio
```

You can get an API key for free by [creating a Cleanlab account](https://tlm.cleanlab.ai/) or experiment with the trustworthiness scores in the [playground](https://chat.cleanlab.ai/chat). Feel free to [email Cleanlab](mailto:suport@cleanlab.ai) with any questions.

Lastly, set the `CLEANLAB_API_KEY` environment variable with the API key.
