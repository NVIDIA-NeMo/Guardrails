# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from pathlib import Path
from typing import Dict, List, Optional

try:
    import redis  # type: ignore
except ImportError:
    redis = None  # type: ignore

from nemoguardrails.rails.llm.config import EmbeddingsCacheConfig

log = logging.getLogger(__name__)


class KeyGenerator(ABC):
    """Abstract class for key generators."""

    name: str  # Class attribute that should be defined in subclasses

    @abstractmethod
    def generate_key(self, text: str) -> str:
        pass

    @classmethod
    def from_name(cls, name):
        for subclass in cls.__subclasses__():
            if subclass.name == name:
                return subclass
        raise ValueError(
            f"Unknown {cls.__name__}: {name}. Available {cls.__name__}s are: "
            f"{', '.join([subclass.name for subclass in cls.__subclasses__()])}"
            ". Make sure to import the derived class before using it."
        )


class HashKeyGenerator(KeyGenerator):
    """Hash-based key generator."""

    name = "hash"

    def generate_key(self, text: str) -> str:
        return str(hash(text))


class MD5KeyGenerator(KeyGenerator):
    """MD5-based key generator."""

    name = "md5"

    def generate_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()


class SHA256KeyGenerator(KeyGenerator):
    """SHA256-based key generator."""

    name = "sha256"

    def generate_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CacheStore(ABC):
    """Abstract class for cache stores."""

    name: str

    @abstractmethod
    def get(self, key):
        """Get a value from the cache."""
        pass

    @abstractmethod
    def set(self, key, value):
        """Set a value in the cache."""
        pass

    @abstractmethod
    def clear(self):
        """Clear the cache."""
        pass

    @classmethod
    def from_name(cls, name):
        for subclass in cls.__subclasses__():
            if subclass.name == name:
                return subclass
        raise ValueError(
            f"Unknown {cls.__name__}: {name}. Available {cls.__name__}s are: "
            f"{', '.join([subclass.name for subclass in cls.__subclasses__()])}"
            ". Make sure to import the derived class before using it."
        )


class InMemoryCacheStore(CacheStore):
    """In-memory cache store.

    This cache store keeps the cache in memory. It does not persist the cache between runs.

    Example:
        >>> cache_store = InMemoryCacheStore()
        >>> cache_store.set('key', 'value')
        >>> print(cache_store.get('key'))  # Outputs: 'value'
        value
    """

    name = "in_memory"

    def __init__(self):
        self._cache = {}

    def get(self, key):
        return self._cache.get(key)

    def set(self, key, value):
        self._cache[key] = value

    def clear(self):
        self._cache = {}


class FilesystemCacheStore(CacheStore):
    """Filesystem cache store.

    This cache store persists the cache between runs by storing it in the filesystem as JSON files.

    Args:
        cache_dir (str, optional): The directory where the cache files will be stored. Defaults to "./cache".

    Example:
        >>> cache_store = FilesystemCacheStore(cache_dir='.cache/embeddings')
        >>> cache_store.set('key', 'value')
        >>> print(cache_store.get('key'))
        value
    """

    name = "filesystem"

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = Path(cache_dir or ".cache/embeddings")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, key):
        return self._cache_dir / str(key)

    def get(self, key):
        file_path = self._get_file_path(key)
        if file_path.exists():
            with open(file_path, "r") as file:
                return json.load(file)
        return None

    def set(self, key, value):
        file_path = self._get_file_path(key)
        with open(file_path, "w") as file:
            json.dump(value, file)

    def clear(self):
        for file_path in self._cache_dir.glob("*"):
            file_path.unlink()


class RedisCacheStore(CacheStore):
    """Redis cache store.

    This cache store keeps the cache in a Redis database. It can be used to share the cache between multiple machines.

    Args:
        redis_client (redis.Redis, optional): The Redis client to use. If not provided, a new client will be created.

    Example:
        >>> redis_client = redis.Redis(host='localhost', port=6379, db=0)
        >>> cache_store = RedisCacheStore(redis_client=redis_client)
        >>> cache_store.set('key', 'value')
        >>> print(cache_store.get('key'))
        value
    """

    name = "redis"

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        if redis is None:
            raise ImportError("Could not import redis, please install it with `pip install redis`.")
        self._redis = redis.Redis(host=host, port=port, db=db)

    def get(self, key):
        return self._redis.get(key)

    def set(self, key, value):
        self._redis.set(key, value)

    def clear(self):
        self._redis.flushall()


class EmbeddingsCache:
    def __init__(
        self,
        key_generator: KeyGenerator,
        cache_store: CacheStore,
        store_config: Optional[dict] = None,
    ):
        self._key_generator = key_generator
        self._cache_store = cache_store
        self._store_config = store_config or {}

    @classmethod
    def from_dict(cls, d: Dict[str, str]):
        key_generator = KeyGenerator.from_name(d.get("key_generator"))()
        store_config_raw = d.get("store_config")
        store_config: dict = store_config_raw if isinstance(store_config_raw, dict) else {}
        cache_store = CacheStore.from_name(d.get("store"))(**store_config)

        return cls(key_generator=key_generator, cache_store=cache_store)

    @classmethod
    def from_config(cls, config: EmbeddingsCacheConfig):
        # config is of type EmbeddingSearchProvider
        return cls.from_dict(config.to_dict())

    def get_config(self):
        return EmbeddingsCacheConfig(
            key_generator=self._key_generator.name,
            store=self._cache_store.name,
            store_config=self._store_config,
        )

    @singledispatchmethod
    def get(self, texts):
        raise NotImplementedError

    @get.register(str)
    def _(self, text: str):
        key = self._key_generator.generate_key(text)
        log.info(f"Fetching key {key} for text '{text[:20]}...' from cache")

        result = self._cache_store.get(key)

        return result

    @get.register(list)
    def _(self, texts: list):
        cached = {}

        for text in texts:
            result = self.get(text)
            if result is not None:
                cached[text] = result

        if len(cached) != len(texts):
            log.info(f"Cache hit rate: {len(cached) / len(texts)}")

        return cached

    @singledispatchmethod
    def set(self, texts):
        raise NotImplementedError

    @set.register(str)
    def _(self, text: str, value: List[float]):
        key = self._key_generator.generate_key(text)
        log.info(f"Cache miss for text '{text}'. Storing key {key} in cache.")
        self._cache_store.set(key, value)

    @set.register(list)
    def _(self, texts: list, values: List[List[float]]):
        for text, value in zip(texts, values):
            self.set(text, value)

    def clear(self):
        self._cache_store.clear()


def cache_embeddings(func):
    """Decorator to cache the embeddings.

    This decorator caches the embeddings in the cache store.
    It uses the `cache_config` attribute of the class to configure the cache.

    If the class does not have a `cache_config` attribute, it will use the `EmbeddingsCacheConfig` by default.
    This decorator can be applied to the `_get_embeddings` method of a subclass of `EmbeddingsIndex` that accepts a list of strings and returns a list of lists of floats.

    Args:
        func (Callable[[Any, List[str]], Awaitable[List[List[float]]]]): The method to decorate. The first argument should be `self`.

    Returns:
        Callable[[Any, List[str]], Awaitable[List[List[float]]]]: The decorated method.

    Example:
        class MyClass:
            @property
            def cache_config(self):
                return EmbeddingsCacheConfig()
            @cache_embeddings
            async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
                # implementation here
                pass
    """

    @functools.wraps(func)
    async def wrapper_decorator(self, texts):
        results = []

        if not self.cache_config.enabled:
            # if cache is not enabled compute embeddings for the whole input
            return await func(self, texts)

        embeddings_cache = EmbeddingsCache.from_config(self.cache_config)

        cached_texts = {}
        uncached_texts = []

        cached_texts = embeddings_cache.get(texts)
        uncached_texts = [text for text in texts if text not in cached_texts]

        # Only call func for uncached texts
        if uncached_texts:
            uncached_results = await func(self, uncached_texts)
            embeddings_cache.set(uncached_texts, uncached_results)

        cached_texts.update(embeddings_cache.get(uncached_texts))
        # Reorder results to match the order of the input texts,
        results = [cached_texts.get(text) for text in texts]
        return results

    return wrapper_decorator
