# Cisco AI Defense Configuration Example

This example contains configuration files for using Cisco AI Defense in your NeMo Guardrails project.

## Files

- **`config.yml`**:  AI Defense configuration with optional settings

## Configuration Options

The AI Defense integration supports configurable timeout and error handling behavior:

- **`timeout`**: API request timeout in seconds (default: 30.0)
- **`fail_open`**: Behavior when API calls fail (default: false for fail closed)

For more details on the Cisco AI Defense integration, see [Cisco AI Defense Integration User Guide](../../../docs/user-guides/community/ai-defense.md).
