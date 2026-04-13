import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from nemoguardrails.llm.clients.base import BaseClient
from nemoguardrails.types import (
    ChatMessage,
    FinishReason,
    LLMResponse,
    LLMResponseChunk,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)

_FINISH_REASON_MAP: Dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}


def _is_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    return (
        name.startswith("o1")
        or name.startswith("o3")
        or (name.startswith("gpt-5") and "chat" not in name)
    )


class OpenAICompatibleClient(BaseClient):
    _ROUTE = "/chat/completions"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        super().__init__(base_url, api_key, timeout)
        self._model = model
        self._default_kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> Optional[str]:
        url = self._base_url.lower()
        if "nvidia" in url or "nim" in url:
            return "nim"
        if "azure" in url:
            return "azure"
        if "openai.com" in url:
            return "openai"
        if "localhost" in url or "127.0.0.1" in url:
            return "local"
        return None

    @property
    def provider_url(self) -> Optional[str]:
        return self._base_url

    def _to_messages(self, prompt: Union[str, List[ChatMessage]]) -> List[Dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        result = []
        for msg in prompt:
            d = msg.to_dict()
            if "tool_calls" in d:
                for tc in d["tool_calls"]:
                    func = tc.get("function", {})
                    if isinstance(func.get("arguments"), dict):
                        func["arguments"] = json.dumps(func["arguments"])
            result.append(d)
        return result

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if stop:
            payload["stop"] = stop
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        merged = {**self._default_kwargs, **kwargs}
        if _is_reasoning_model(self._model):
            merged.pop("temperature", None)
            merged.pop("stop", None)
            payload.pop("stop", None)
        payload.update(merged)
        return payload

    async def generate_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = self._to_messages(prompt)
        payload = self._build_payload(messages, stop=stop, **kwargs)
        data = await self._apost(self._ROUTE, payload)
        return self._parse_response(data)

    async def stream_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponseChunk]:
        messages = self._to_messages(prompt)
        payload = self._build_payload(messages, stop=stop, stream=True, **kwargs)
        async for chunk_data in self._apost_stream(self._ROUTE, payload):
            chunk = self._parse_chunk(chunk_data)
            if chunk is not None:
                yield chunk

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        choice = data["choices"][0]
        message = choice.get("message", {})

        content = message.get("content") or ""
        reasoning = message.get("reasoning_content")

        tool_calls = None
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            tool_calls = [self._parse_tool_call(tc) for tc in raw_tool_calls]

        raw_finish = choice.get("finish_reason")
        finish_reason = _FINISH_REASON_MAP.get(raw_finish, "other") if raw_finish else None

        usage = None
        raw_usage = data.get("usage")
        if raw_usage:
            input_tokens = raw_usage.get("prompt_tokens", 0)
            output_tokens = raw_usage.get("completion_tokens", 0)
            usage = UsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=raw_usage.get("total_tokens") or (input_tokens + output_tokens),
                reasoning_tokens=raw_usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            )

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            model=data.get("model"),
            finish_reason=finish_reason,
            usage=usage,
        )

    def _parse_chunk(self, data: Dict[str, Any]) -> Optional[LLMResponseChunk]:
        choices = data.get("choices", [])
        if not choices:
            raw_usage = data.get("usage")
            if raw_usage:
                input_tokens = raw_usage.get("prompt_tokens", 0)
                output_tokens = raw_usage.get("completion_tokens", 0)
                return LLMResponseChunk(
                    usage=UsageInfo(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=raw_usage.get("total_tokens") or (input_tokens + output_tokens),
                    ),
                )
            return None

        choice = choices[0]
        delta = choice.get("delta", {})

        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        raw_finish = choice.get("finish_reason")
        finish_reason = _FINISH_REASON_MAP.get(raw_finish, "other") if raw_finish else None

        return LLMResponseChunk(
            delta_content=content,
            delta_reasoning=reasoning,
            model=data.get("model"),
            finish_reason=finish_reason,
        )

    @staticmethod
    def _parse_tool_call(tc: Dict[str, Any]) -> ToolCall:
        func = tc.get("function", {})
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args_dict = json.loads(raw_args)
            except json.JSONDecodeError:
                args_dict = {}
        else:
            args_dict = raw_args

        return ToolCall(
            id=tc["id"],
            type=tc.get("type", "function"),
            function=ToolCallFunction(
                name=func.get("name", ""),
                arguments=args_dict,
            ),
        )
