# Common Integrations

## Jaeger

```bash
docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
```

## Zipkin

```bash
docker run -d -p 9411:9411 openzipkin/zipkin
pip install opentelemetry-exporter-zipkin
```

## OpenTelemetry Collector

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  batch:

exporters:
  logging:

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [logging]
```
