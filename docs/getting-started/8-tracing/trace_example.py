# trace_example.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from nemoguardrails import LLMRails, RailsConfig

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource



# Configure OpenTelemetry
resource = Resource.create({"service.name": "guardrails-hero-parallel"})
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))



config = RailsConfig.from_path("configs/parallel")
rails = LLMRails(config)
response = rails.generate(messages=[{"role": "user", "content": "Hello!"}])
print(f"Response: {response}")