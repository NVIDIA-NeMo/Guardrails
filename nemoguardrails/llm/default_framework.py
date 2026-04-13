import os
from typing import Any, Dict, List, Optional

from nemoguardrails.types import LLMModel

_DEFAULT_BASE_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "nim": "https://integrate.api.nvidia.com/v1",
    "nvidia_ai_endpoints": "https://integrate.api.nvidia.com/v1",
    "ollama": "http://localhost:11434/v1",
}

_API_KEY_ENV_VARS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "nim": "NVIDIA_API_KEY",
    "nvidia_ai_endpoints": "NVIDIA_API_KEY",
}


def _resolve_base_url(provider_name: str) -> str:
    url = _DEFAULT_BASE_URLS.get(provider_name)
    if url:
        return url
    raise ValueError(
        f"No default base_url for provider '{provider_name}'. "
        "Set it explicitly in model parameters: parameters.base_url"
    )


def _resolve_api_key(provider_name: str) -> Optional[str]:
    env_var = _API_KEY_ENV_VARS.get(provider_name)
    if env_var:
        return os.environ.get(env_var)
    return None


class DefaultFramework:
    def __init__(self):
        self._providers: Dict[str, Any] = {}

    def create_model(
        self,
        model_name: str,
        provider_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LLMModel:
        kwargs = dict(model_kwargs) if model_kwargs else {}
        kwargs.pop("mode", None)

        if provider_name in self._providers:
            return self._providers[provider_name](model=model_name, **kwargs)

        from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient

        base_url = kwargs.pop("base_url", None) or _resolve_base_url(provider_name)
        api_key = kwargs.pop("api_key", None) or _resolve_api_key(provider_name)

        return OpenAICompatibleClient(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )

    def register_provider(self, name: str, provider_cls: Any) -> None:
        self._providers[name] = provider_cls

    def get_provider_names(self) -> List[str]:
        return sorted(set(list(_DEFAULT_BASE_URLS.keys()) + list(self._providers.keys())))
