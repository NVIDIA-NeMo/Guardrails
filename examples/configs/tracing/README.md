# README

We encourage you to implement a log adapter for the production environment based on your specific requirements.

To use the `FileSystem` and `OpenTelemetry` adapters, please install the following dependencies:

```bash
pip install opentelemetry-api opentelemetry-sdk aiofiles
```

If you want to use Zipkin as a backend, you can use the following command to start a Zipkin server:

1. Install the Zipkin exporter for OpenTelemetry:

    ```sh
    pip install opentelemetry-exporter-zipkin
    ```

2. Run the `Zipkin` server using Docker:

    ```sh
    docker run -d -p 9411:9411 openzipkin/zipkin
    ```

3. Update the `config.yml` to set the exporter to Zipkin:

    ```yaml
    tracing:
      enabled: true
      adapters:
        - name: OpenTelemetry
          service_name: "nemo_guardrails_service"
          exporter: "zipkin"
          resource_attributes:
            env: "production"
