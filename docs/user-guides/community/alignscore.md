# AlignScore Integration

NeMo Guardrails provides out-of-the-box support for the [AlignScore metric (Zha et al.)](https://aclanthology.org/2023.acl-long.634.pdf), which uses a RoBERTa-based model for scoring factual consistency in model responses with respect to the knowledge base.

In our testing, we observed an average latency of ~220ms on hosting AlignScore as an HTTP service, and ~45ms on direct inference with the model loaded in-memory. This makes it much faster than the self-check method. However, this method requires an on-prem deployment of the publicly available AlignScore model. Please see the [AlignScore Deployment](../advanced/align-score-deployment.md) guide for more details.

## Usage

To use the AlignScore-based fact-checking, you have to set the following configuration options in your `config.yml`:

```yaml
rails:
  config:
    fact_checking:
      parameters:
        # Point to a running instance of the AlignScore server
        endpoint: "http://localhost:5000/alignscore_large"

  output:
    flows:
      - alignscore check facts
```

The Colang flow for AlignScore-based fact-checking rail is the same as that for the self-check fact-checking rail. To trigger the fact-checking rail, you have to set the `$check_facts` context variable to `True` before a bot message that requires fact-checking, e.g.:

```colang
define flow
  user ask about report
  $check_facts = True
  bot provide report answer
```
