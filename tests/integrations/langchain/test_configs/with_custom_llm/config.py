from nemoguardrails.llm.providers import register_llm_provider
from tests.integrations.langchain.test_configs.with_custom_llm.custom_llm import CustomLLM

register_llm_provider("custom_llm", CustomLLM)
