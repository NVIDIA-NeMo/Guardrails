# Tracing Configuration

NeMo Guardrails includes a tracing feature that allows you to monitor and log interactions for better observability and debugging. Tracing can be easily configured via the existing `config.yml` file. Below are the steps to enable and configure tracing in your project.

## Enabling Tracing

To enable tracing, set the enabled flag to true under the tracing section in your `config.yml`:

```yaml
tracing:
  enabled: true
```

```{important}
You must install the necessary dependencies to use tracing adapters.

```sh
  pip install "opentelemetry-api opentelemetry-sdk aiofiles"
```

## Configuring Tracing Adapters

Tracing supports multiple adapters that determine how and where the interaction logs are exported. You can configure one or more adapters by specifying them under the adapters list. Below are examples of configuring the built-in `OpenTelemetry` and `FileSystem` adapters:

```yaml
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
      service_name: "nemo_guardrails_service"
      exporter: "console"  # Options: "console", "zipkin", etc.
      resource_attributes:
        env: "production"
    - name: FileSystem
      filepath: './traces/traces.jsonl'
```

```{warning}
The "console" is intended for debugging and demonstration purposes only and should not be used in production environments. Using this exporter will output tracing information directly to the console, which can interfere with application output, distort the user interface, degrade performance, and potentially expose sensitive information. For production use, please configure a suitable exporter that sends tracing data to a dedicated backend or monitoring system.
```

### OpenTelemetry Adapter

The `OpenTelemetry` adapter integrates with the OpenTelemetry framework, allowing you to export traces to various backends. Key configuration options include:

 • `service_name`: The name of your service.
 • `exporter`: The type of exporter to use (e.g., console, zipkin).
 • `resource_attributes`: Additional attributes to include in the trace resource (e.g., environment).

### FileSystem Adapter

The  `FileSystem`  adapter exports interaction logs to a local JSON Lines file. Key configuration options include:

 • `filepath`: The path to the file where traces will be stored. If not specified, it defaults to `./.traces/trace.jsonl`.

## Example Configuration

Below is a comprehensive example of a `config.yml` file with both `OpenTelemetry` and `FileSystem` adapters enabled:

```yaml
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
      service_name: "nemo_guardrails_service"
      exporter: "zipkin"
      resource_attributes:
        env: "production"
    - name: FileSystem
      filepath: './traces/traces.jsonl'
```

To use this configuration, you must ensure that Zipkin is running locally or is accessible via the network.

### Using Zipkin as an Exporter

To use `Zipkin` as an exporter, follow these steps:

1. Install the Zipkin exporter for OpenTelemetry:

    ```sh
    pip install opentelemetry-exporter-zipkin
    ```

2. Run the `Zipkin` server using Docker:

    ```sh
    docker run -d -p 9411:9411 openzipkin/zipkin
    ```

## Registering OpenTelemetry Exporters

You can also use other [OpenTelemetry exporters](https://opentelemetry.io/ecosystem/registry/?component=exporter&language=python) by registering them in the `config.py` file. To do so you need to use `register_otel_exporter` and register the exporter class.Below is an example of registering the `Jaeger` exporter:

```python
# This assumes that Jaeger exporter is installed
# pip install opentelemetry-exporter-jaeger

from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from nemoguardrails.tracing.adapters.opentelemetry import register_otel_exporter

register_otel_exporter(JaegerExporter, "jaeger")

  ```

Then you can use it in the `config.yml` file as follows:

```yaml

tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
      service_name: "nemo_guardrails_service"
      exporter: "jaeger"
      resource_attributes:
        env: "production"

```

## Custom InteractionLogAdapters

NeMo Guardrails allows you to extend its tracing capabilities by creating custom `InteractionLogAdapter` classes. This flexibility enables you to transform and export interaction logs to any backend or format that suits your needs.

### Implementing a Custom Adapter

To create a custom adapter, you need to implement the `InteractionLogAdapter` abstract base class. Below is the interface you must follow:

```python
from abc import ABC, abstractmethod
from nemoguardrails.tracing import InteractionLog

class InteractionLogAdapter(ABC):
    name: Optional[str] = None


    @abstractmethod
    async def transform_async(self, interaction_log: InteractionLog):
        """Transforms the InteractionLog into the backend-specific format asynchronously."""
        raise NotImplementedError

    async def close(self):
        """Placeholder for any cleanup actions if needed."""
        pass

    async def __aenter__(self):
        """Enter the runtime context related to this object."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        await self.close()

```

### Registering Your Custom Adapter

After implementing your custom adapter, you need to register it so that NemoGuardrails can recognize and utilize it. This is done by adding a registration call in your `config.py:`

```python
from nemoguardrails.tracing.adapters.registry import register_log_adapter
from path.to.your.adapter import YourCustomAdapter

register_log_adapter(YourCustomAdapter, "CustomLogAdapter")
```

### Example: Creating a Custom Adapter

Here's a simple example of a custom adapter that logs interaction logs to a custom backend:

```python
from nemoguardrails.tracing.adapters.base import InteractionLogAdapter
from nemoguardrails.tracing import InteractionLog

class MyCustomLogAdapter(InteractionLogAdapter):
    name = "MyCustomLogAdapter"

    def __init__(self, custom_option1: str, custom_option2: str):
      self.custom_option1 = custom_option1
      self.custom_option2 = custom

    def transform(self, interaction_log: InteractionLog):
        # Implement your transformation logic here
        custom_format = convert_to_custom_format(interaction_log)
        send_to_custom_backend(custom_format)

    async def transform_async(self, interaction_log: InteractionLog):
        # Implement your asynchronous transformation logic here
        custom_format = convert_to_custom_format(interaction_log)
        await send_to_custom_backend_async(custom_format)

    async def close(self):
        # Implement any necessary cleanup here
        await cleanup_custom_resources()

```

Updating `config.yml` with Your `CustomLogAdapter`

Once registered, you can configure your custom adapter in the `config.yml` like any other adapter:

```yaml
tracing:
  enabled: true
  adapters:
    - name: MyCustomLogAdapter
      custom_option1: "value1"
      custom_option2: "value2"

```

By following these steps, you can leverage the built-in tracing adapters or create and integrate your own custom adapters to enhance the observability of your NeMo Guardrails powered applications. Whether you choose to export logs to the filesystem, integrate with OpenTelemetry, or implement a bespoke logging solution, tracing provides the flexibility to meet your requirements.
