from typing import Any, AsyncIterator, List, Optional, Union

from nemoguardrails.types import ChatMessage, LLMResponse, LLMResponseChunk


class CustomLLM:
    def __init__(self, model: str = "custom_llm", **kwargs: Any):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> Optional[str]:
        return "custom_llm"

    @property
    def provider_url(self) -> Optional[str]:
        return None

    async def generate_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="Custom LLM response")

    async def stream_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponseChunk]:
        yield LLMResponseChunk(delta_content="Custom LLM response")
