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

import typer
from dataset_tools import Banking77Connector, ChitChatConnector

app = typer.Typer()


@app.command()
def main(
    dataset_name: str = typer.Option(
        default="banking",
        exists=True,
        help="Name of the dataset. Possible values: banking, chitchat.",
    ),
    dataset_path: str = typer.Option(
        default="./banking/original-data",
        help="Path to the dataset data, can be downloaded from original repos.",
    ),
    max_samples_intent: int = typer.Option(
        default=0,
        help="Maximum samples per intent. If value is 0, use all samples.",
    ),
):
    """
    Create the user.co files needed to run the topical evaluation and Guardrails app
    for the banking and chit-chat datasets used in the evaluation experiments.

    This code can be easily adapted for any other public chatbot dataset.
    """

    print(
        f"Creating user.co file for {dataset_name} dataset. "
        f"Path: {dataset_path} , max samples per intent: {max_samples_intent}"
    )

    if dataset_name == "banking":
        dataset = Banking77Connector()
        dataset.read_dataset()
        dataset.write_colang_output(
            output_file_name="./banking/user.co",
            num_samples_per_intent=max_samples_intent,
        )
        print("Created user.co file for banking dataset.")
    elif dataset_name == "chitchat":
        dataset = ChitChatConnector()
        dataset.read_dataset()
        dataset.write_colang_output(
            output_file_name="./chitchat/user.co",
            num_samples_per_intent=max_samples_intent,
        )
        print("Created user.co file for chitchat dataset.")
    else:
        print(f"Unknown dataset {dataset_name}, cannot create user.co file!")


if __name__ == "__main__":
    app()
