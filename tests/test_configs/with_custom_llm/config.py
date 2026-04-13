from nemoguardrails.llm.providers import register_provider
from tests.test_configs.with_custom_llm.custom_llm import CustomLLM

register_provider("custom_llm", CustomLLM)
