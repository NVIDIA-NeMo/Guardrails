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

import os
from typing import Tuple

import numpy as np

SNOWFLAKE_MODEL_ID = "Snowflake/snowflake-arctic-embed-m-long"


class SnowflakeEmbed:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = os.environ.get("JAILBREAK_CHECK_DEVICE")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            SNOWFLAKE_MODEL_ID,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            SNOWFLAKE_MODEL_ID,
            trust_remote_code=True,
            add_pooling_layer=False,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, text: str):
        tokens = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=2048)
        tokens = tokens.to(self.device)
        embeddings = self.model(**tokens)[0][:, 0]
        return embeddings.detach().cpu().squeeze(0).numpy()


class JailbreakClassifier:
    def __init__(self, random_forest_path: str):
        from onnxruntime import InferenceSession

        self.embed = SnowflakeEmbed()
        # See https://onnx.ai/sklearn-onnx/auto_examples/plot_convert_decision_function.html
        self.classifier = InferenceSession(random_forest_path, providers=["CPUExecutionProvider"])

    def __call__(self, text: str) -> Tuple[bool, float]:
        e = self.embed(text)
        x = np.asarray([e], dtype=np.float32)
        res = self.classifier.run(None, {"X": x})
        classification = res[0].item()
        # The second is a list of dicts of probabilities -- the slice res[1][:2] should have only one element.
        # We access the dict entry for the class.
        prob = res[1][0][classification]
        score = -prob if classification == 0 else prob
        return bool(classification), float(score)
