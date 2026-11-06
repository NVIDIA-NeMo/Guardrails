# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Automatic windows for local text classifiers without truncation."""

import threading
from dataclasses import dataclass
from typing import Any

from transformers import BatchEncoding, TextClassificationPipeline
from transformers.pipelines.base import ChunkPipeline
from transformers.tokenization_utils_base import LARGE_INTEGER

from nemoguardrails.library.hf_classifier.compatibility import ensure_overflow_support


@dataclass
class _PreparedInput:
    """Validated tokenization that can be passed to HF data-loader workers."""

    encoding: BatchEncoding
    is_chunked: bool = False

    @property
    def window_count(self) -> int:
        return len(self.encoding["input_ids"]) if self.is_chunked else 1


def _content_tokens(encoding: Any) -> list[int]:
    return [
        token
        for token, special, attended in zip(encoding.ids, encoding.special_tokens_mask, encoding.attention_mask)
        if not special and attended
    ]


def _verify_coverage(expected: list[int], encoded: BatchEncoding, stride: int, direction: str) -> None:
    """Reject missing or changed content before any window reaches inference."""
    if direction == "left":
        expected = expected[::-1]
    covered = 0
    for index, encoding in enumerate(encoded.encodings or []):
        content = _content_tokens(encoding)
        if direction == "left":
            content.reverse()
        start = covered - stride if index else 0
        end = start + len(content)
        if end <= covered or content != expected[start:end]:
            break
        covered = end
    else:
        if covered == len(expected):
            return
    raise ValueError(
        "HF classifier windows do not cover the complete input token sequence. "
        "Use a compatible tokenizer version; tokenizers 0.23.1 and 0.23.2 lose overflow tokens."
    )


def _input_limit(pipe: Any, max_length: int | None = None) -> int | None:
    model_limit = getattr(pipe.model.config, "max_position_embeddings", None)
    # RoBERTa-style position embeddings reserve positions through padding_idx.
    embeddings = getattr(pipe.model.base_model, "embeddings", None)
    positions = getattr(embeddings, "position_embeddings", None)
    padding_idx = getattr(positions, "padding_idx", None)
    if model_limit is not None and padding_idx is not None:
        model_limit -= padding_idx + 1
    # A smaller tokenizer limit can be a preprocessing preference, not a model constraint.
    return next(
        (
            value
            for value in (model_limit, pipe.tokenizer.model_max_length, max_length)
            if value is not None and 0 < value < LARGE_INTEGER
        ),
        None,
    )


class ChunkedTextClassificationPipeline(ChunkPipeline, TextClassificationPipeline):
    """Reuse HF tokenization, inference and scoring for oversized text inputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inference_lock = threading.Lock()

    def _sanitize_parameters(self, **kwargs):
        has_max_windows = "max_windows" in kwargs
        max_windows = kwargs.pop("max_windows", None)
        preprocess, forward, postprocess = super()._sanitize_parameters(**kwargs)
        if has_max_windows:
            if type(max_windows) is not int or max_windows < 1:
                raise ValueError("HF classifier max_windows must be a positive integer.")
            preprocess["max_windows"] = max_windows
        return preprocess, forward, postprocess

    def __call__(self, inputs, **kwargs):
        # Fast tokenizers mutate truncation settings; cached pipelines are shared by executor threads.
        with self._inference_lock:
            return super().__call__(inputs, **kwargs)

    def __getstate__(self):
        # Spawned HF data-loader workers pickle the pipeline, but thread locks cannot be pickled.
        state = self.__dict__.copy()
        state.pop("_inference_lock")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._inference_lock = threading.Lock()

    def get_iterator(self, inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params):
        if batch_size > 1 or num_workers:
            # Native scalar inference never constructs a batch-padding collator.
            if isinstance(inputs, list) and len(inputs) == 1:
                prepared = self._prepare_input(inputs[0], **preprocess_params)
                if prepared.window_count == 1:
                    first = next(self.preprocess(prepared))
                    yield self.postprocess([self.forward(first, **forward_params)], **postprocess_params)
                    return
                inputs = [prepared]
            yield from super().get_iterator(
                inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params
            )
            return
        # Score immediately instead of retaining every window's model outputs.
        for text in inputs:
            outputs: Any = (
                self.forward(window, **forward_params) for window in self.preprocess(text, **preprocess_params)
            )
            yield self.postprocess(outputs, **postprocess_params)

    def preprocess(self, inputs, **kwargs):
        prepared = inputs if isinstance(inputs, _PreparedInput) else self._prepare_input(inputs, **kwargs)
        if not prepared.is_chunked:
            yield {**prepared.encoding, "is_last": True}
            return
        for index in range(prepared.window_count):
            # Convert one window at a time; default inference needs no padding token.
            window = BatchEncoding({key: [value[index]] for key, value in prepared.encoding.items()}, tensor_type="pt")
            yield {**window, "is_last": index == prepared.window_count - 1}

    def _prepare_input(self, inputs, max_windows=128, **kwargs) -> _PreparedInput:
        """Preserve native truncation, otherwise verify and bound overlapping windows."""
        assert self.tokenizer is not None
        prepared: Any = super().preprocess(inputs, **kwargs)
        limit = _input_limit(self, kwargs.get("max_length"))
        truncation = kwargs.get("truncation")
        truncates = truncation not in (None, False, "do_not_truncate") or (
            truncation is None and kwargs.get("max_length") is not None and kwargs.get("padding", False) is False
        )
        if truncates or limit is None or prepared["input_ids"].shape[-1] <= limit:
            return _PreparedInput(prepared)
        if not self.tokenizer.is_fast:
            raise ValueError("Oversized HF classifier inputs require a fast tokenizer for overlapping windows.")
        ensure_overflow_support()
        expected = _content_tokens(prepared.encodings[0])
        del prepared
        special_tokens = (
            self.tokenizer.num_special_tokens_to_add(pair=False) if kwargs.get("add_special_tokens", True) else 0
        )
        capacity = limit - special_tokens
        if capacity <= 0:
            raise ValueError("HF classifier window limit must exceed the tokenizer's special-token count.")
        stride = kwargs.get("stride", min(32, capacity // 4))
        if not 0 <= stride < capacity:
            raise ValueError(
                "HF classifier stride must be non-negative and smaller than the window's content token capacity."
            )
        step = capacity - stride
        window_count = 1 + max(0, (len(expected) - capacity + step - 1) // step)
        if window_count > max_windows:
            raise ValueError(
                "HF classifier input exceeds max_windows; shorten the input or increase the window budget."
            )
        encoded = self.tokenizer(
            inputs,
            **{**kwargs, "max_length": limit, "stride": stride, "truncation": True, "return_overflowing_tokens": True},
        )
        encoded.pop("overflow_to_sample_mapping", None)
        count = len(encoded["input_ids"])
        if count > max_windows:
            raise ValueError(
                "HF classifier input exceeds max_windows; shorten the input or increase the window budget."
            )
        _verify_coverage(expected, encoded, stride, self.tokenizer.truncation_side)
        if any(len(ids) > limit for ids in encoded["input_ids"]):
            raise ValueError("HF classifier padding exceeds the model input limit; adjust the padding parameters.")
        return _PreparedInput(encoded, is_chunked=True)

    def _forward(self, model_inputs):
        is_last = model_inputs.pop("is_last")
        return {**super()._forward(model_inputs), "is_last": is_last}

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        results = []
        for output in model_outputs:
            results.append(
                super().postprocess(output, function_to_apply=function_to_apply, top_k=top_k, _legacy=_legacy)
            )
        if len(results) == 1:
            return results[0]
        return [item for result in results for item in (result if isinstance(result, list) else [result])]
