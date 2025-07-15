# Clavata Integration

[Clavata](https://clavata.ai) provides real-time moderation capabilities allowing anyone to detect and filter content. The exact rules of what to filter are up to you, but we do provide a number of rulesets for common issues.

This integration enables NeMo Guardrails to use Clavata for content moderation, topic moderation, and dialog moderation in both input and output flows.

## Getting Access

To sign up for Clavata or obtain an API key:

- [Request access](https://www.clavata.ai/) through the website
- Contact support at <hello@clavata.ai>

## Setup

1. Ensure you have access to the Clavata platform and have configured your content moderation policies. You'll need:
   - Your Clavata API key
   - Policy IDs for the content types you want to moderate
   - (Optional) A custom server endpoint if provided by Clavata.ai

2. Set the `CLAVATA_API_KEY` environment variable with your Clavata API key:

   ```bash
   export CLAVATA_API_KEY="your-api-key"
   ```

3. Configure your `config.yml` according to the following example:

```yaml
rails:
  config:
    clavata:
      policies:
        Threats: 00000000-0000-0000-0000-000000000000
        Toxicity: 00000000-0000-0000-0000-000000000000
      label_match_logic: ALL  # "ALL" | "ANY"
      input:
        # Reference an alias above in `policies`
        policy: Threats
      output:
        policy: Toxicity
        # Optional: Specify labels to require specific matches
        labels:
          - Hate Speech
          - Self-harm
      # Optional: Only provide this if you've been told to by Clavata.ai
      server_endpoint: "https://some-alt-endpoint.com"
  # Optional: reference the built-in flows
  input:
    flows:
      - clavata check input
  output:
    flows:
      - clavata check output
```

## Configuration Details

- `server_endpoint`: The Clavata API endpoint (only if provided by Clavata.ai)
- `policies`: Map of policy aliases to each policy's unique ID in your Clavata.ai account
- `label_match_logic`: (Optional) `ALL` requires all labels specified for a rail to match, `ANY` requires at least one match. Defaults to `ANY` if not set.
- `input/output`: Flow-specific configurations
  - `policy`: The policy alias to use for this flow
  - `labels`: (Optional) List of specific labels to check for

## Usage

The Clavata integration provides two ways to implement content moderation:

### 1. Built-in Flows

#### For users of Colang 1.0

Add these flows to your configuration to automatically check content when using _Colang 1.0_:

```yaml
rails:
  input:
    flows:
      - clavata check input  # Check user input
  output:
    flows:
      - clavata check output  # Check LLM output
```

#### For users of Colang 2.0

If you're using Colang 2.0, there's no need to specify configuration for input and output rails in your `config.yml`. In fact, doing so is now deprecated. The good news is that because Colang 2.0 supports flows with variables, you can specify which policy to use (and even which labels to match) inline in the definitions for any of your rails (i.e., input, output, dialog, etc.)

Here's an example of how to configure an input rail to check against a specific Clavata policy:

```colang
import guardrails
import nemoguardrails.library.clavata


# Check the input against the "Toxicity" policy
flow input rails $input_text
    clavata check for ($input_text, Toxicity)

# To make the check even more strict so it only matches particular labels in the policy, you can add a comma-separated list of labels at the end:
flow input rails $input_text
    clavata check for ($input_text, Toxicity, ["Hate Speech","Harassment"])
```

> The same is true for `output` flows, of course. See [our example](../../../examples/configs/clavata_v2/rails.co) for more.

### 2. Programmatic Usage

If you are using colang 2.x, you can make use of the Clavata action in your own flows:

```colang
# Check content
$is_match = await ClavataCheckAction(text=$some_text, policy=$some_policy_alias)
```

The action returns `True` if the content matches the specified policy's criteria.

## Customization

You can customize the content moderation behavior by:

1. Configuring different policies for input and output flows
2. Specifying which labels must match within a policy
3. Setting the label match logic to either "ALL" (all specified labels must match) or "ANY" (at least one label must match)

## Error Handling

If the Clavata API request fails, the system will raise a `ClavataPluginAPIError`. The integration will also raise a `ClavataPluginValueError` if there are configuration issues, such as:

- Invalid policy aliases
- Missing required configuration
- Invalid flow types

## Notes

- Ensure that your Clavata API key is properly set up and accessible
- The integration currently supports content moderation checks for input and output flows
- You can configure different policies and label requirements for input and output flows
- If no labels are specified for a policy, any label match will be considered a hit

For more information on Clavata and its capabilities, please refer to the [Clavata documentation](https://clavata.helpscoutdocs.com).
