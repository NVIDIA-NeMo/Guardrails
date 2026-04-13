from nemoguardrails.llm.providers import register_provider
from tests.test_configs.with_custom_chat_model.custom_chat_model import CustomChatModel

register_provider("custom_chat_model", CustomChatModel)
