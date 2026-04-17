import importlib.util
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

import pytest

from guardrails.llm_providers import (
    ArbitraryCallable,
    AsyncArbitraryCallable,
    LLMResponse,
    PromptCallableException,
    chat_prompt,
    get_async_llm_ask,
    get_llm_ask,
)
from guardrails.utils.safe_get import safe_get_with_brackets

from .mocks import MockAsyncOpenAILlm, MockOpenAILlm


def test_openai_callable_does_not_retry_on_success(mocker):
    llm = MockOpenAILlm()
    succeed_spy = mocker.spy(llm, "succeed")

    arbitrary_callable = ArbitraryCallable(
        llm.succeed, messages=[{"role": "user", "content": "Hello"}]
    )
    response = arbitrary_callable()

    assert succeed_spy.call_count == 1
    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.asyncio
async def test_async_openai_callable_does_not_retry_on_success(mocker):
    llm = MockAsyncOpenAILlm()
    succeed_spy = mocker.spy(llm, "succeed")

    arbitrary_callable = AsyncArbitraryCallable(
        llm.succeed, messages=[{"role": "user", "content": "Hello"}]
    )
    response = await arbitrary_callable()

    assert succeed_spy.call_count == 1
    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.fixture(scope="module")
def openai_chat_mock():
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    return ChatCompletion(
        id="",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Mocked LLM output",
                    role="assistant",
                ),
            ),
        ],
        created=0,
        model="",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=20,
            prompt_tokens=10,
            total_tokens=30,
        ),
    )


@pytest.fixture(scope="module")
def openai_chat_stream_mock():
    def gen():
        # Returns a generator object
        for i in range(4, 8):
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"{i},"},
                        "finish_reason": None,
                    }
                ]
            }

    return gen()


@pytest.fixture(scope="module")
def openai_mock():
    @dataclass
    class MockCompletionUsage:
        completion_tokens: int
        prompt_tokens: int
        total_tokens: int

    @dataclass
    class MockCompletionChoice:
        finish_reason: str
        index: int
        logprobs: Any
        text: str

    @dataclass
    class MockCompletion:
        id: str
        choices: List[MockCompletionChoice]
        created: int
        model: str
        object: str
        usage: MockCompletionUsage

    return MockCompletion(
        id="",
        choices=[
            MockCompletionChoice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                text="Mocked LLM output",
            ),
        ],
        created=0,
        model="",
        object="text_completion",
        usage=MockCompletionUsage(
            completion_tokens=20,
            prompt_tokens=10,
            total_tokens=30,
        ),
    )


@pytest.fixture(scope="module")
def openai_stream_mock():
    def gen():
        # Returns a generator object
        for i in range(4, 8):
            yield {
                "choices": [{"text": f"{i},", "finish_reason": None}],
                "model": "openai-model-name",
            }

    return gen()


@pytest.mark.skipif(
    not importlib.util.find_spec("manifest"),
    reason="manifest-ml is not installed",
)
def test_manifest_callable():
    client = MagicMock()
    client.run.return_value = "Hello world!"

    from guardrails.llm_providers import ManifestCallable

    manifest_callable = ManifestCallable()
    response = manifest_callable(text="Hello", client=client)

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.skipif(
    not importlib.util.find_spec("manifest"),
    reason="manifest-ml is not installed",
)
@pytest.mark.asyncio
async def test_async_manifest_callable():
    client = MagicMock()

    async def return_async():
        return ["Hello world!"]

    client.arun_batch.return_value = return_async()

    from guardrails.llm_providers import AsyncManifestCallable

    manifest_callable = AsyncManifestCallable()
    response = await manifest_callable(text="Hello", client=client)

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
@pytest.mark.parametrize(
    "model_inputs,tokenizer_call_count", [(None, 1), ({"input_ids": ["Hello"]}, 0)]
)
def test_hugging_face_model_callable(mocker, model_inputs, tokenizer_call_count):
    class MockTokenizer:
        def __call__(self, prompt: str, *args: Any, **kwds: Any) -> Dict[str, Any]:
            self.prompt = prompt
            return self

        def to(self, *args, **kwargs):
            return {"input_ids": [self.prompt]}

        def decode(self, output: str, *args, **kwargs) -> str:
            return output

    tokenizer = MockTokenizer()

    tokenizer_call_spy = mocker.spy(tokenizer, "to")
    tokenizer_decode_spy = mocker.spy(tokenizer, "decode")

    model_generate = MagicMock()
    model_generate.return_value = ["Hello there!"]

    from guardrails.llm_providers import HuggingFaceModelCallable

    hf_model_callable = HuggingFaceModelCallable()
    response = hf_model_callable(
        model_generate=model_generate,
        messages=[{"role": "user", "content": "Hello"}],
        tokenizer=tokenizer,
    )

    assert tokenizer_call_spy.call_count == 1
    assert tokenizer_decode_spy.call_count == 1
    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello there!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
def test_hugging_face_pipeline_callable():
    pipeline = MagicMock()
    pipeline.return_value = [{"generated_text": "Hello there!"}]

    from guardrails.llm_providers import HuggingFacePipelineCallable

    hf_model_callable = HuggingFacePipelineCallable()
    response = hf_model_callable(
        pipeline=pipeline, messages=[{"role": "user", "content": "Hello"}]
    )

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello there!"
    assert response.prompt_token_count is None
    assert response.response_token_count is None


@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
def test_litellm_callable(mocker):
    # Mock the litellm.completion function and
    # the classes it returns
    @dataclass
    class Message:
        content: str

    @dataclass
    class Choice:
        message: Message

    @dataclass
    class Usage:
        prompt_tokens: int
        completion_tokens: int

    @dataclass
    class MockResponse:
        choices: List[Choice]
        usage: Usage

    class MockCompletion:
        @staticmethod
        def create() -> MockResponse:
            return MockResponse(
                choices=[Choice(message=Message(content="Hello there!"))],
                usage=Usage(prompt_tokens=10, completion_tokens=20),
            )

    mocker.patch("litellm.completion", return_value=MockCompletion.create())

    from guardrails.llm_providers import LiteLLMCallable

    litellm_callable = LiteLLMCallable()
    response = litellm_callable("Hello")

    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello there!"
    assert response.prompt_token_count == 10
    assert response.response_token_count == 20


def test_minimax_callable(mocker):
    """Test MiniMaxCallable wraps MiniMax OpenAI-compatible API correctly."""
    from dataclasses import dataclass as _dc

    @_dc
    class _Message:
        content: str

    @_dc
    class _Choice:
        message: _Message

    @_dc
    class _Usage:
        prompt_tokens: int
        completion_tokens: int

    @_dc
    class _MockResponse:
        choices: List[_Choice]
        usage: _Usage

    mock_response = _MockResponse(
        choices=[_Choice(message=_Message(content="MiniMax says hello!"))],
        usage=_Usage(prompt_tokens=5, completion_tokens=10),
    )

    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("openai.Client", return_value=mock_client)

    mocker.patch.dict("os.environ", {"MINIMAX_API_KEY": "test-minimax-key"})

    from guardrails.llm_providers import MiniMaxCallable

    callable_ = MiniMaxCallable()
    response = callable_(
        text="Hello",
        model="MiniMax-M2.7",
    )

    assert isinstance(response, LLMResponse)
    assert response.output == "MiniMax says hello!"
    assert response.prompt_token_count == 5
    assert response.response_token_count == 10

    # Verify temperature was set to 1.0 (MiniMax requires > 0)
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 1.0


def test_minimax_callable_uses_custom_base_url(mocker):
    """Test MiniMaxCallable uses custom base_url when provided."""
    from dataclasses import dataclass as _dc

    @_dc
    class _Message:
        content: str

    @_dc
    class _Choice:
        message: _Message

    @_dc
    class _MockResponse:
        choices: List[_Choice]
        usage: Any

    mock_response = _MockResponse(
        choices=[_Choice(message=_Message(content="response"))],
        usage=None,
    )
    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    captured = {}

    def mock_openai_client(api_key=None, base_url=None):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return mock_client

    mocker.patch("openai.Client", side_effect=mock_openai_client)
    mocker.patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"})

    from guardrails.llm_providers import MiniMaxCallable

    callable_ = MiniMaxCallable()
    callable_(
        text="Hello",
        model="MiniMax-M2.7",
        base_url="https://custom.minimax.io/v1",
    )

    assert captured["base_url"] == "https://custom.minimax.io/v1"


def test_minimax_callable_raises_without_api_key(mocker):
    """Test MiniMaxCallable raises when no API key is available."""
    mocker.patch.dict("os.environ", {}, clear=True)
    # Make sure MINIMAX_API_KEY is not set
    mocker.patch.dict("os.environ", {"MINIMAX_API_KEY": ""})

    from guardrails.llm_providers import MiniMaxCallable, PromptCallableException

    callable_ = MiniMaxCallable()

    with pytest.raises(PromptCallableException):
        callable_(text="Hello", model="MiniMax-M2.7")


def test_get_llm_ask_minimax_model():
    """Test that model names starting with 'MiniMax' route to MiniMaxCallable."""
    import os

    os.environ["MINIMAX_API_KEY"] = "test-key"
    try:
        from guardrails.llm_providers import MiniMaxCallable

        result = get_llm_ask(None, model="MiniMax-M2.7")
        assert isinstance(result, MiniMaxCallable)
    finally:
        del os.environ["MINIMAX_API_KEY"]


def test_get_llm_ask_minimax_highspeed_model():
    """Test that MiniMax-M2.7-highspeed also routes to MiniMaxCallable."""
    import os

    os.environ["MINIMAX_API_KEY"] = "test-key"
    try:
        from guardrails.llm_providers import MiniMaxCallable

        result = get_llm_ask(None, model="MiniMax-M2.7-highspeed")
        assert isinstance(result, MiniMaxCallable)
    finally:
        del os.environ["MINIMAX_API_KEY"]


def test_get_llm_ask_minimax_temperature_not_set_to_zero():
    """Test that MiniMax models skip the default temperature=0 warning."""
    import warnings as _warnings

    import os

    os.environ["MINIMAX_API_KEY"] = "test-key"
    try:
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            result = get_llm_ask(None, model="MiniMax-M2.7")
            # Should not emit the temperature deprecation warning for MiniMax
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
    finally:
        del os.environ["MINIMAX_API_KEY"]


def test_get_llm_ask_returns_prompt_callable_base_directly():
    """Test that PromptCallableBase instances are returned directly."""
    from guardrails.llm_providers import MiniMaxCallable

    instance = MiniMaxCallable()
    result = get_llm_ask(instance, model="MiniMax-M2.7")
    assert result is instance


class ReturnTempCallable(Callable):
    def __call__(self, *args, messages=None, **kwargs) -> Any:
        return ""


@pytest.mark.parametrize(
    "llm_api, args, kwargs, expected_temperature",
    [
        (ReturnTempCallable(), [], {"temperature": 0.5}, 0.5),
        (ReturnTempCallable(), [], {}, 0),
        (ReturnTempCallable(), [], {"model": "gpt-5-nano"}, None),
    ],
)
def test_get_llm_ask_temperature(llm_api, args, kwargs, expected_temperature):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = get_llm_ask(llm_api, *args, **kwargs)
        if expected_temperature is None:
            assert "temperature" not in result.init_kwargs
            assert len(w) == 0
        else:
            assert "temperature" in result.init_kwargs
            assert result.init_kwargs["temperature"] == expected_temperature
            if "temperature" not in kwargs:
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "default value of 0 for temperature is deprecated" in str(
                    w[0].message
                )


@pytest.mark.skipif(
    not importlib.util.find_spec("manifest"),
    reason="manifest is not installed",
)
def test_get_llm_ask_manifest(mocker):
    def mock_os_environ_get(key, *args):
        if key == "OPENAI_API_KEY":
            return "sk-xxxxxxxxxxxxxx"
        return safe_get_with_brackets(os.environ, key, *args)

    mocker.patch("os.environ.get", side_effect=mock_os_environ_get)

    from manifest import Manifest

    from guardrails.llm_providers import ManifestCallable

    manifest_client = Manifest("openai")

    prompt_callable = get_llm_ask(manifest_client)

    assert isinstance(prompt_callable, ManifestCallable)


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers"),
    reason="transformers is not installed",
)
def test_get_llm_ask_hugging_face_model(mocker):
    from transformers import PreTrainedModel, GenerationMixin

    from guardrails.llm_providers import HuggingFaceModelCallable

    class MockModel(PreTrainedModel, GenerationMixin):
        _modules: Any

        def __init__(self, *args, **kwargs):
            self._modules = {}

    mock_model = MockModel()

    prompt_callable = get_llm_ask(mock_model.generate)

    assert isinstance(prompt_callable, HuggingFaceModelCallable)


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers"),
    reason="transformers is not installed",
)
def test_get_llm_ask_hugging_face_pipeline():
    from transformers import Pipeline

    from guardrails.llm_providers import HuggingFacePipelineCallable

    class MockPipeline(Pipeline):
        task = "text-generation"

        def __init__(self, *args, **kwargs):
            pass

        def _forward():
            pass

        def _sanitize_parameters():
            pass

        def postprocess():
            pass

        def preprocess():
            pass

    mock_pipeline = MockPipeline()

    prompt_callable = get_llm_ask(mock_pipeline)

    assert isinstance(prompt_callable, HuggingFacePipelineCallable)


@pytest.mark.skipif(
    not importlib.util.find_spec("litellm"),
    reason="`litellm` is not installed",
)
def test_get_llm_ask_litellm():
    from litellm import completion

    from guardrails.llm_providers import LiteLLMCallable

    prompt_callable = get_llm_ask(completion)

    assert isinstance(prompt_callable, LiteLLMCallable)


def test_get_llm_ask_custom_llm():
    from guardrails.llm_providers import ArbitraryCallable

    def my_llm(prompt: str, *, messages=None, **kwargs) -> str:
        return f"Hello {prompt}!"

    prompt_callable = get_llm_ask(my_llm)

    assert isinstance(prompt_callable, ArbitraryCallable)


def test_get_llm_ask_custom_llm_warning():
    from guardrails.llm_providers import ArbitraryCallable

    def my_llm(prompt: str, **kwargs) -> str:
        return f"Hello {prompt}!"

    with pytest.warns(
        UserWarning,
        match=(
            "We recommend including 'messages'"
            " as keyword-only arguments for custom LLM callables."
            " Doing so ensures these arguments are not unintentionally"
            " passed through to other calls via \\*\\*kwargs."
        ),
    ):
        prompt_callable = get_llm_ask(my_llm)

        assert isinstance(prompt_callable, ArbitraryCallable)


def test_get_llm_ask_custom_llm_must_accept_kwargs():
    def my_llm(messages: str) -> str:
        return f"Hello {messages}!"

    with pytest.raises(
        ValueError, match="Custom LLM callables must accept \\*\\*kwargs!"
    ):
        get_llm_ask(my_llm)


def test_get_async_llm_ask_custom_llm():
    from guardrails.llm_providers import AsyncArbitraryCallable

    async def my_llm(messages: str, **kwargs) -> str:
        return f"Hello {messages}!"

    prompt_callable = get_async_llm_ask(my_llm)

    assert isinstance(prompt_callable, AsyncArbitraryCallable)


def test_get_async_llm_ask_custom_llm_warning():
    from guardrails.llm_providers import AsyncArbitraryCallable

    async def my_llm(**kwargs) -> str:
        return "Hello world!"

    with pytest.warns(
        UserWarning,
        match=(
            "We recommend including 'messages'"
            " as keyword-only arguments for custom LLM callables."
            " Doing so ensures these arguments are not unintentionally"
            " passed through to other calls via \\*\\*kwargs."
        ),
    ):
        prompt_callable = get_async_llm_ask(my_llm)

        assert isinstance(prompt_callable, AsyncArbitraryCallable)


def test_get_async_llm_ask_custom_llm_must_accept_kwargs():
    def my_llm(prompt: str) -> str:
        return f"Hello {prompt}!"

    with pytest.raises(
        ValueError, match="Custom LLM callables must accept \\*\\*kwargs!"
    ):
        get_async_llm_ask(my_llm)


def test_chat_prompt():
    # raises when messages are not provided
    with pytest.raises(PromptCallableException):
        chat_prompt(None)
