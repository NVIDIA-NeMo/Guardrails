# HiddenLayer AI Runtime Security Example

This example checks user input, bot output, and the calls to and results from the `run_sql` tool with HiddenLayer AI Runtime Security.

Set the credentials, and set `project_id` in `config.yml` to your HiddenLayer project ID or alias:

```bash
export HIDDENLAYER_CLIENT_ID=...
export HIDDENLAYER_CLIENT_SECRET=...
export OPENAI_API_KEY=...
```

The tool rails run on the IORails engine, which the `Guardrails` class selects for this configuration:

```python
from nemoguardrails import Guardrails, RailsConfig

config = RailsConfig.from_path("examples/configs/hiddenlayer")
rails = Guardrails(config)
```

The `run_sql` tool is declared on the main model, so the per-tool rails can validate its calls.
Refer to the HiddenLayer page in the guardrail catalog for the full configuration reference.
