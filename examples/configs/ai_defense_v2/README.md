# Cisco AI Defense Configuration Example (Colang 2.x)

This example contains configuration files for using Cisco AI Defense with Colang 2.x in your NeMo Guardrails project.

## Files

- **`config.yaml`**: AI Defense configuration with optional settings
- **`main.co`**: Main flow definition
- **`rails.co`**: Input and output rails definitions for AI Defense

## Configuration Options

The AI Defense integration supports configurable timeout and error handling behavior:

- **`timeout`**: API request timeout in seconds (default: 30.0)
- **`fail_open`**: Behavior when API calls fail (default: false for fail closed)
  - `false`: Fail closed - blocks content when API errors occur
  - `true`: Fail open - allows content when API errors occur


## Environment Variables

Before running this example, set the required environment variables:

```bash
export AI_DEFENSE_API_KEY="your-api-key"
export AI_DEFENSE_API_ENDPOINT="us.api.inspect.aidefense.security.cisco.com/api/v1/inspect/chat"
```

For more details on the Cisco AI Defense integration, see [Cisco AI Defense Integration User Guide](../../../docs/user-guides/community/ai-defense.md).
