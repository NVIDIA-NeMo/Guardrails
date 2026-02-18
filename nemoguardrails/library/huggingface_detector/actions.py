# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from typing import Dict, Optional

from nemoguardrails.actions.actions import action
from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)

# Cache for loaded models to avoid reloading on every call
_model_cache: Dict[str, tuple] = {}


def _load_model_and_tokenizer(model_repo: str, device: Optional[str] = None):
    """Load and cache a Huggingface model and tokenizer.

    Args:
        model_repo: Huggingface model repository ID (e.g., 'ibm-granite/granite-guardian-hap-38m')
        device: Device to load the model onto (e.g., 'cuda', 'cpu', 'cuda:0')

    Returns:
        Tuple of (model, tokenizer)
    """
    cache_key = f"{model_repo}:{device}" if device else model_repo

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        log.info(f"Loading Huggingface model: {model_repo}" + (f" on device: {device}" if device else ""))
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        model = AutoModelForSequenceClassification.from_pretrained(model_repo)

        # Move model to specified device if provided
        if device:
            model = model.to(device)

        _model_cache[cache_key] = (model, tokenizer)
        log.info(f"Successfully loaded model: {model_repo}" + (f" on device: {device}" if device else ""))

        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load Huggingface model '{model_repo}': {str(e)}") from e


def _get_label_dicts(model) -> [Dict, Dict]:
    """Convert label2id and id2label if one is missing.

    If both are present, this is a no-op
    If one is present, create the other and return both
    If neither are present, return both as None
    """
    label2id = getattr(model.config, "label2id", None)
    id2label = getattr(model.config, "id2label", None)

    if label2id is None and id2label is not None:
        label2id = {label: idx for idx, label in id2label.items()}
    elif id2label is None and label2id is not None:
        id2label = {idx: label for label, idx in label2id.items()}
    elif label2id is None and id2label is None:
        pass  # leave them none for handling later

    return id2label, label2id


def _standardize_blocked_classes_to_indices(blocked_classes, label2id) -> set:
    """Convert blocked classes (labels or indices) from the HuggingfaceModelConfig to a set of indices.

    Args:
        blocked_classes: List of class labels (strings) or indices (integers)
        label2id: Dictionary mapping class labels to indices, if provided by the model or is inferrable

    Returns:
        Set of integer indices corresponding to blocked classes

    Raises:
        ValueError: If a class label is not found in the model's label mapping
            or if labels are provided but the model has no label mapping
    """
    if not blocked_classes:
        return set()

    # Check if we have labels (strings) or indices (integers)
    if isinstance(blocked_classes[0], str):
        if label2id is None:
            raise ValueError(
                "Model does not provide label mappings (id2label or label2id). "
                "Please use class indices instead of labels in blocked_classes configuration. "
                "Example: blocked_classes: [0, 1, 2]"
            )

        blocked_indices = set()
        for label in blocked_classes:
            if label not in label2id:
                available_labels = list(label2id.keys())
                raise ValueError(
                    f"Class label '{label}' not found in model's label mapping. Available labels: {available_labels}"
                )
            blocked_indices.add(label2id[label])

        return blocked_indices
    else:
        # Already indices, just convert to set
        return set(blocked_classes)


def _classify_text(text: str, model, tokenizer) -> tuple:
    """Classify text using the loaded model.

    Args:
        text: Input text to classify
        model: Loaded Huggingface model
        tokenizer: Loaded tokenizer

    Returns:
        Tuple of (predicted_index, score_list) where:
            - predicted_index: Integer index of the top prediction
            - score_list: List of scores indexed by class index
    """
    import torch

    # Tokenize and run inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted index
    predicted_index = probs[0].argmax().item()

    # Get scores as a list
    score_list = probs[0].tolist()

    return predicted_index, score_list


def _check_model(text: str, model_config, model_repo: str) -> dict:
    """Check text against a model.

    Args:
        text: Text to classify
        model_config: HuggingfaceModelConfig instance
        model_repo: Model repository ID (for caching)

    Returns:
        Dictionary with classification results
    """
    # Load model and tokenizer
    model, tokenizer = _load_model_and_tokenizer(model_repo, device=model_config.device)

    # Retrieve (if needed, construct) the model's id2label and label2id mappings
    id2label, label2id = _get_label_dicts(model)

    # if class names were used in the blocked classes config, standardize to use indices
    blocked_indices = _standardize_blocked_classes_to_indices(model_config.blocked_classes, label2id)

    # Classify text
    predicted_index, score_list = _classify_text(text, model, tokenizer)

    # Get predicted score
    predicted_score = score_list[predicted_index]

    # Get predicted label for logging and output (use index if no label mapping)
    if id2label:
        predicted_label = id2label.get(predicted_index, str(predicted_index))
    else:
        predicted_label = str(predicted_index)

    # Check if predicted class is in blocked list
    is_blocked = predicted_index in blocked_indices

    # Build all_scores dict with labels (or indices if no labels available)
    all_scores = {}
    for idx, score in enumerate(score_list):
        if id2label:
            key = id2label.get(idx, str(idx))
        else:
            key = str(idx)
        all_scores[key] = score

    return {
        "model_repo": model_repo,
        "detected_class": predicted_label,
        "score": predicted_score,
        "all_scores": all_scores,
        "blocked": is_blocked,
    }


@action(output_mapping=lambda result: not result.get("blocked", True))
async def huggingface_detector_check(
    model_repo: str = None,
    text: str = None,
    rail_type: str = None,
    config: RailsConfig = None,
    **kwargs,
) -> dict:
    """Function to check text using a Huggingface *text classification, predictive* model.

    Args:
        model_repo: Huggingface model repository ID to use for checking
        text: Text to check
        config: Rails configuration
        rail_type: Whether this is rail is an input, output, tool_input, etc.

    Returns:
        Dictionary with:
            - allowed: bool indicating if text is allowed
            - detected_class: str with the predicted class label
            - score: float with the confidence score
            - all_scores: dict with all class scores
            - model_repo: str with the model repository ID
            - descriptor: optional str with model descriptor
    """
    if not text:
        raise ValueError("No text provided for Huggingface detector check")

    if not model_repo:
        raise ValueError(
            "model_repo parameter is required. "
            "Please specify which model to use, e.g., model_repo='ibm-granite/granite-guardian-hap-38m'"
        )

    # Get configuration
    if config is None or config.rails.config.huggingface_detector is None:
        raise ValueError(
            "Huggingface detector configuration is required. "
            "Please configure 'rails.config.huggingface_detector' in config.yml"
        )

    hf_config = config.rails.config.huggingface_detector

    # Find the specified model in the configuration
    model_config = None
    for mc in hf_config.models:
        if mc.model_repo == model_repo:
            model_config = mc
            break

    if model_config is None:
        raise ValueError(
            f"The colang flow requested model '{model_repo}', "
            f"but his model is not present in the huggingface_detector configuration. "
            f"Available models: {[m.model_repo for m in hf_config.models]}"
        )

    if not model_config.blocked_classes:
        log.warning(f"No blocked classes configured for model {model_repo}. All {rail_type} messages will be allowed.")
        return {
            "allowed": True,
            "detected_class": None,
            "score": 0.0,
            "all_scores": {},
            "model_repo": model_repo,
            "descriptor": model_config.descriptor,
        }

    # Check text against the specified model
    result = _check_model(text, model_config, model_repo)
    is_allowed = not result["blocked"]

    # Build model identifier for logging
    model_identifier = f"{model_repo}"
    if model_config.descriptor:
        model_identifier += f" ({model_config.descriptor})"

    log.info(
        f"Huggingface detector {rail_type} [{model_identifier}]: text classified as "
        f"'{result['detected_class']}' (score: {result['score']:.4f}), "
        f"allowed: {is_allowed}"
    )

    return {
        "allowed": is_allowed,
        "detected_class": result["detected_class"],
        "score": result["score"],
        "all_scores": result["all_scores"],
        "model_repo": model_repo,
        "descriptor": model_config.descriptor,
    }
