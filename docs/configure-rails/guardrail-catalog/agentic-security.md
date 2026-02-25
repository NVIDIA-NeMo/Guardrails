---
title:
  page: "Agentic Security"
  nav: "Agentic Security"
description: "Reference for agentic security guardrails that protect LLM-based agents using tools and interacting with external systems."
topics: ["Configuration", "AI Safety", "Security"]
tags: ["Rails", "Injection Detection", "Agentic", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Agentic Security

Agentic security provides specialized guardrails for LLM-based agents that use tools and interact with external systems.

## Injection Detection

The NeMo Guardrails library offers detection of potential exploitation attempts by using injection such as code injection, cross-site scripting, SQL injection, and template injection.
Injection detection is primarily intended to be used in agentic systems to enhance other security controls as part of a defense-in-depth strategy.

The first part of injection detection is [YARA rules](https://yara.readthedocs.io/en/stable/index.html).
A YARA rule specifies a set of strings (text or binary patterns) to match and a Boolean expression that specifies the logic of the rule.
YARA rules are a technology that is familiar to many security teams.

The second part of injection detection is specifying the action to take when a rule is triggered.
You can specify to *reject* the text and return "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of {detections}."
Rejecting the output is the safest action and most appropriate for production deployments.
As an alternative to rejecting the output, you can specify to *omit* the triggering text from the response.

### About the Default Rules

By default, the NeMo Guardrails library provides the following rules:

- Code injection (Python): Recommended if the LLM output is used as an argument to downstream functions or passed to a code interpreter.
- SQL injection: Recommended if the LLM output is used as part of a SQL query to a database.
- Template injection (Jinja): Recommended for use if LLM output is rendered using the Jinja templating language.
  This rule is usually paired with code injection rules.
- Cross-site scripting (Markdown and Javascript): Recommended if the LLM output is rendered directly in HTML or Markdown.

You can view the default rules in the [yara_rules directory](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/nemoguardrails/library/injection_detection/yara_rules) of the GitHub repository.

### Configuring Injection Detection

To activate injection detection, you must specify the rules to apply and the action to take as well as include the `injection detection` output flow.
As an example config:

```yaml
rails:
  config:
    injection_detection:
      injections:
        - code
        - sqli
        - template
        - xss
      action:
        reject

  output:
    flows:
      - injection detection
```

Refer to the following table for the `rails.config.injection_detection` field syntax reference:

```{list-table}
:header-rows: 1

* - Field
  - Description
  - Default Value

* - `injections`
  - Specifies the injection detection rules to use.
    The following injections are part of the library:

    - `code` for Python code injection
    - `sqli` for SQL injection
    - `template` for Jinja template injection
    - `xss` for cross-site scripting
  - None (required)

* - `action`
  - Specifies the action to take when injection is detected.
    Refer to the following actions:

    - `reject` returns a message to the user indicating that the query could not be handled and they should try again.
    - `omit` returns the model response, removing the offending detected content.
  - None (required)

* - `yara_path`
  - Specifies the path to a directory that contains custom YARA rules.
  - `library/injection_detection/yara_rules` in the NeMo Guardrails package.

* - `yara_rules`
  - Specifies inline YARA rules.
    The field is a dictionary that maps rule names to the rules.
    The rules use the string data type.

    ```yaml
    yara_rules:
      <inline-rule-name>: |-
        <inline-rule-content>
    ```

    If specified, these inline rules override the rules found in the `yara_path` field.
  - None
```

For information about writing YARA rules, refer to the [YARA documentation](https://yara.readthedocs.io/en/stable/index.html).

### Example

Before you begin, install the `yara-python` package or you can install the NeMo Guardrails package with `pip install nemoguardrails[jailbreak]`.

1. Set your NVIDIA API key as an environment variable:

   ```console
   $ export NVIDIA_API_KEY=<nvapi-...>
   ```

1. Create a configuration directory, such as `config`, and add a `config.yml` file with contents like the following:

   ```{literalinclude} ../../../examples/configs/injection_detection/config/config.yml
   :language: yaml
   ```

1. Load the guardrails configuration:

   ```{literalinclude} ../../../examples/configs/injection_detection/demo.py
   :language: python
   :start-after: "# start-load-config"
   :end-before: "# end-load-config"
   ```

1. Send a possibly unsafe request:

   ```{literalinclude} ../../../examples/configs/injection_detection/demo.py
   :language: python
   :start-after: "# start-unsafe-response"
   :end-before: "# end-unsafe-response"
   ```

   *Example Output*

   ```{literalinclude} ../../../examples/configs/injection_detection/demo-out.txt
   :start-after: "# start-unsafe-response"
   :end-before: "# end-unsafe-response"
   ```
