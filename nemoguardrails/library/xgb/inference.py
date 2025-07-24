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

import logging
import pickle
from typing import List

log = logging.getLogger(__name__)
MODEL_REGISTRY = {
    "SPAM": {
        "model_path": "nemoguardrails/library/xgb/model_artifacts/model.pkl",
        "vectorizer_path": "nemoguardrails/library/xgb/model_artifacts/vectorizer.pkl",
    }
}


def xgb_inference(text: str, enabled_detectors: List[str]):
    detections = []
    for detector in enabled_detectors:
        model_info = MODEL_REGISTRY.get(detector)
        if not model_info:
            raise ValueError(
                f"XGB detector '{detector}' is not configured in the MODEL_REGISTRY."
            )
        model_path = model_info["model_path"]
        vectorizer_path = model_info["vectorizer_path"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        try:
            X_vec = vectorizer.transform([text])
            prediction = model.predict(X_vec)[0]
            probability = model.predict_proba(X_vec)[0]

            is_safe = prediction == 0
            confidence = max(probability)

            detections.append(
                {
                    "allowed": bool(is_safe),
                    "score": float(confidence),
                    "prediction": "safe" if is_safe else detector,
                }
            )

        except Exception as e:
            raise ValueError(
                f"Error during XGBoost inference for detector '{detector}': {e}"
            )
    return detections
