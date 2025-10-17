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

from typing import Tuple

import numpy as np


class SnowflakeEmbed:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "snowflake/snowflake-arctic-embed-m-long"
        )
        self.model = AutoModel.from_pretrained(
            "snowflake/snowflake-arctic-embed-m-long",
            trust_remote_code=True,
            add_pooling_layer=False,
            safe_serialization=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, text: str):
        tokens = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt", max_length=2048
        )
        tokens = tokens.to(self.device)
        embeddings = self.model(**tokens)[0][:, 0]
        return embeddings.detach().cpu().squeeze(0).numpy()


class JailbreakClassifier:
    def __init__(self, random_forest_path: str):
        import pickle

        self.embed = SnowflakeEmbed()
        with open(random_forest_path, "rb") as fd:
            self.classifier = pickle.load(fd)

    def __call__(self, text: str) -> Tuple[bool, float]:
        e = self.embed(text)
        probs = self.classifier.predict_proba([e])
        classification = np.argmax(probs)
        prob = np.max(probs)
        score = -prob if classification == 0 else prob
        return bool(classification), float(score)
