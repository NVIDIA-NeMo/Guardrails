import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

log = logging.getLogger(__name__)


class BaseClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ImportError(
                "httpx is required for the default framework. "
                "Install it with: pip install httpx"
            )

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _apost(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = await self._get_client()
        response = await client.post(
            f"{self._base_url}{path}",
            json=payload,
            headers=self._build_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def _apost_stream(
        self, path: str, payload: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        client = await self._get_client()
        async with client.stream(
            "POST",
            f"{self._base_url}{path}",
            json=payload,
            headers=self._build_headers(),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    log.warning("Failed to parse SSE chunk: %s", data)

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
