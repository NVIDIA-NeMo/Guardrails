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

import os
from functools import lru_cache
from typing import List

import nltk
import typer
import uvicorn
from alignscore import AlignScore
from fastapi import FastAPI
from pydantic import BaseModel

# Make sure we have the punkt tokenizer downloaded.
nltk.download("punkt")

models_path = os.environ.get("ALIGN_SCORE_PATH")

if models_path is None:
    raise ValueError(
        "Please set the ALIGN_SCORE_PATH environment variable "
        "to point to the AlignScore checkpoints folder. "
    )

app = FastAPI()

device = os.environ.get("ALIGN_SCORE_DEVICE", "cpu")


@lru_cache
def get_model(model: str):
    """Initialize a model.

    Args
        model: The type of the model to be loaded, i.e. "base", "large".
    """
    return AlignScore(
        model="roberta-base",
        batch_size=32,
        device=device,
        ckpt_path=os.path.join(models_path, f"AlignScore-{model}.ckpt"),
        evaluation_mode="nli_sp",
    )


class AlignScoreRequest(BaseModel):
    evidence: str
    claim: str


@app.get("/")
def hello_world():
    welcome_str = (
        f"This is a development server to host AlignScore models.\n"
        + f"<br>Hit the /alignscore_base or alignscore_large endpoints with "
        f"a POST request containing evidence and claim.\n"
        + f"<br>Example: curl -X POST -d 'evidence=This is an evidence "
        f"passage&claim=This is a claim.' http://localhost:8000/alignscore_base\n"
    )
    return welcome_str


def get_alignscore(model, evidence: str, claim: str) -> dict:
    return {"alignscore": model.score(contexts=[evidence], claims=[claim])[0]}


@app.post("/alignscore_base")
def alignscore_base(request: AlignScoreRequest):
    model = get_model("base")
    return get_alignscore(model, request.evidence, request.claim)


@app.post("/alignscore_large")
def alignscore_large(request: AlignScoreRequest):
    model = get_model("large")
    return get_alignscore(model, request.evidence, request.claim)


cli_app = typer.Typer()


@cli_app.command()
def start(
    port: int = typer.Option(
        default=5000, help="The port that the server should listen on. "
    ),
    models: List[str] = typer.Option(
        default=["base"],
        help="The list of models to be loaded on startup",
    ),
    initialize_only: bool = typer.Option(
        default=False, help="Whether to run only the initialization for the models."
    ),
):
    # Preload the models
    for model in models:
        typer.echo(f"Pre-loading model {model}.")
        get_model(model)

    if initialize_only:
        print("Initialization successful.")
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    cli_app()
