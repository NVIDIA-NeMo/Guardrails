# QA Started

This is a QA started guide for using automated tests to test all the examples in `nemoguardrails/examples/_deprecated` folder. This guide will cover:

1. Software installation prerequisites;
2. Running automated tests;
3. Analyzing logs if there are any failure(s).

## Installation

1. Install scikit-learn for validating bot response.

 ```bash
 > pip install -U scikit-learn
 ```

## Running automated tests to test all the examples in `nemoguardrails/examples/_deprecated` folder.

Please refer to the [installation guide](installation-guide.md) for instructions on how to install the NeMo Guardrails library.

1. Make sure that you have the `OPENAI_API_KEY` and `WOLFRAM_ALPHA_APP_ID` environment variables set.

 ```bash
 > export OPENAI_API_KEY=...
 > export WOLFRAM_ALPHA_APP_ID=...
 ```

2. Change the directory to `nemoguardrails/qa` folder, and then run all the automated tests

 ```bash
 > QA=True python -m pytest test_*.py
 ```

NOTE: The QA tests are skipped by default as they are expensive (i.e., they make live call to OpenAI and other services). To enable them, you have to set the `QA` environment variable.

Alternatively, you can also run the tests from the root of the project:

```bash
> QA=True pytest qa
```

3. If there are any failure(s), analyze the corresponding example test log.

 ```bash
 > ls -l *.log
 > moderation_rail.log
 > execution_rails.log
 > grounding_rail.log
 > jailbreak_check.log
 > topical_rail.log
 ```
