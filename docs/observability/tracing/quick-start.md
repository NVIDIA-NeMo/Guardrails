# Quick Start

The following is a minimal setup to enable tracing using the OpenTelemetry SDK.

1. Install the NeMo Guardrails toolkit and the OpenTelemetry SDK.

    ```bash
    pip install nemoguardrails[tracing] opentelemetry-sdk
    ```

2. Set up tracing as follows and save as `trace_example.py`.

    ```python
    # trace_example.py
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from nemoguardrails import LLMRails, RailsConfig

    # Configure OpenTelemetry
    resource = Resource.create({"service.name": "guardrails-quickstart"})
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Configure guardrails with tracing
    config_yaml = """
    models:
      - type: main
        engine: openai
        model: gpt-4o-mini

    rails:
      config:
        streaming: true

    tracing:
      enabled: true
      adapters:
        - name: OpenTelemetry
    """

    config = RailsConfig.from_content(yaml_content=config_yaml)
    rails = LLMRails(config)
    response = rails.generate(messages=[{"role": "user", "content": "Hello!"}])
    print(f"Response: {response}")
    ```

3. Run the script:

    ```bash
    python trace_example.py
    ```
