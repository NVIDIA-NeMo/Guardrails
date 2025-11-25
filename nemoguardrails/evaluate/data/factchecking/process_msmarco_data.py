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

import json

import pandas as pd
import tqdm
from datasets import load_dataset

# Install the datasets library using pip install datasets
# Load the dataset
dataset = load_dataset("ms_marco", "v2.1")

# Use the validation split and convert to pandas dataframe
df = pd.DataFrame(dataset["validation"])

# Convert the dataframe to a json file with "question", "answers" and "evidence" as keys
fact_check_data = []

for idx, row in tqdm.tqdm(df.iterrows()):
    sample = {}
    sample["question"] = row["query"]
    sample["answer"] = row["answers"][0]
    if row["passages"]["is_selected"].count(1) == 1:
        sample["evidence"] = row["passages"]["passage_text"][row["passages"]["is_selected"].index(1)]
        fact_check_data.append(sample)

# Save the json file
with open("msmarco.json", "w") as f:
    json.dump(fact_check_data, f)
