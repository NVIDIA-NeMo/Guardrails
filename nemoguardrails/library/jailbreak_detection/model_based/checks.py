# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from functools import lru_cache
from pathlib import Path

from nemoguardrails.library.jailbreak_detection.model_based.models import (
    JailbreakClassifier,
)

models_path = os.environ.get("EMBEDDING_CLASSIFIER_PATH")

# When we add NIM support, will need to remove this check.
if models_path is None:
    raise EnvironmentError(
        "Please set the EMBEDDING_CLASSIFIER_PATH environment variable to point to the Classifier model_based folder"
    )


@lru_cache()
def initialize_model(classifier_path: str = models_path) -> JailbreakClassifier:
    """
    Initialize the global classifier model according to the configuration provided.
    Args
        classifier_path: Path to the classifier model
    Returns
        jailbreak_classifier: JailbreakClassifier object combining embedding model and NemoGuard JailbreakDetect RF
    """

    jailbreak_classifier = JailbreakClassifier(str(Path(classifier_path).joinpath("snowflake.pkl")))

    return jailbreak_classifier


def check_jailbreak(
    prompt: str,
    classifier: JailbreakClassifier = None,
) -> dict:
    """
    Use embedding-based jailbreak detection model to check for the presence of a jailbreak
    Args:
        prompt: User utterance to classify
        classifier: Instantiated JailbreakClassifier object
    """
    if classifier is None:
        classifier = initialize_model()

    classification, score = classifier(prompt)
    # classification will be 1 or 0 -- cast to boolean.
    return {"jailbreak": classification, "score": score}
