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
import os.path
import pickle
from pathlib import Path
from typing import Optional

import faiss
import pandas as pd
import torch
from gpt4pandas import GPT4Pandas

try:
    from langchain.chains import RetrievalQA
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import FAISS
except ImportError:
    try:
        from langchain_classic.chains import RetrievalQA
        from langchain_classic.embeddings import HuggingFaceEmbeddings
        from langchain_classic.text_splitter import CharacterTextSplitter
        from langchain_classic.vectorstores import FAISS
    except ImportError as e:
        raise ImportError(
            "Failed to import required LangChain modules. "
            "For LangChain >=1.0.0, install langchain-classic: pip install langchain-classic"
        ) from e
from langchain_core.language_models import BaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import register_llm_provider
from nemoguardrails.llm.providers.huggingface import HuggingFacePipelineCompatible

from .tabular_llm import TabularLLM


def _get_model_config(config: RailsConfig, type: str):
    """Quick helper to return the config for a specific model type."""
    for model_config in config.models:
        if model_config.type == type:
            return model_config


def _load_model(model_name_or_path, device, num_gpus, debug=False):
    """Load an HF locally saved checkpoint."""
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update(
                    {
                        "device_map": "auto",
                        "max_memory": {i: "13GiB" for i in range(num_gpus)},
                    }
                )
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        print("mps not supported")
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, **kwargs)

    if device == "cuda" and num_gpus == 1:
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


def _make_faiss_gpu(data_path, out_path, embeddings):
    # Here we process the txt files under the data_path folder.
    ps = list(Path(data_path).glob("**/*.txt"))
    print(ps)
    data = []
    sources = []
    for p in ps:
        with open(p) as f:
            data.append(f.read())
        sources.append(p)

    # We do this due to the context limits of the LLMs.
    # Here we split the documents, as needed, into smaller chunks.
    # We do this due to the context limits of the LLMs.
    text_splitter = CharacterTextSplitter(chunk_size=200, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(data):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": sources[i]}] * len(splits))

    # Here we create a vector store from the documents and save it to disk.
    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    os.makedirs(out_path, exist_ok=True)
    faiss.write_index(store.index, out_path + "docs.index")
    store.index = None
    with open(out_path + "faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
    return store


def _get_vector_db(model_name: str, data_path: str, persist_path: str):
    """Creates a vector DB for a given data path.

    If it's the first time, the index will be persisted at the given path.
    Otherwise, it will be loaded directly (if the `persist_path` exists).
    """
    # use other embeddings from huggingface
    model_kwargs = {"device": "cuda"}

    hf_embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    using_vectorstore = "faiss"
    if using_vectorstore == "faiss":
        if os.path.exists(persist_path):
            index = faiss.read_index(os.path.join(persist_path, "docs.index"))
            with open(os.path.join(persist_path, "faiss_store.pkl"), "rb") as f:
                vectordb = pickle.load(f)
            vectordb.index = index
        else:
            data_path = data_path
            vectordb = _make_faiss_gpu(data_path, persist_path, hf_embedding)
    return vectordb


def init_main_llm(config: RailsConfig):
    """Initialize the main model from a locally saved path.

    The path is taken from the main model config.

    models:
      - type: main
        engine: hf_pipeline_bloke
        parameters:
          path: "<PATH TO THE LOCALLY SAVED CHECKPOINT>"
    """
    # loading custom llm  from disk with multiGPUs support
    # model_name = "< path_to_the_saved_custom_llm_checkpoints >"  # loading model ckpt from disk
    model_config = _get_model_config(config, "main")
    model_path = model_config.parameters.get("path")
    device = model_config.parameters.get("device", "cuda")
    num_gpus = model_config.parameters.get("num_gpus", 1)
    model, tokenizer = _load_model(model_path, device, num_gpus, debug=False)

    # repo_id="TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
    # pipe = pipeline("text-generation", model=repo_id, device_map={"":"cuda:0"}, max_new_tokens=256, temperature=0.1, do_sample=True,use_cache=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
    )

    hf_llm = HuggingFacePipelineCompatible(pipeline=pipe)
    provider = get_llm_instance_wrapper(llm_instance=hf_llm, llm_type="hf_pipeline_bloke")
    register_llm_provider("hf_pipeline_bloke", provider)


def _get_titanic_raw_data_frame(csv_path: str):
    """Reads the Titanic CSV file and returns a tweaked data frame."""
    df = pd.read_csv(csv_path, sep=",")

    # working on the data
    Embarked_d = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    class_d = {1: "first class", 2: "second class", 3: "third class"}
    df["Class"] = df["Pclass"].apply(lambda x: class_d[x])

    # changing the embark port to full name
    n = len(df)
    col_ls = list(df.columns)
    idx = col_ls.index("Embarked")
    ls = []
    for i in range(n):
        temp = df.iloc[i, idx]
        if isinstance(temp, str):
            out = Embarked_d[temp]
            ls.append(out)
        else:
            ls.append("N/A")

    df["port"] = ls
    df["Lived"] = df["Survived"].apply(lambda x: "survived" if x == 1 else "died")

    # dropping duplicated and re-worked column
    df.drop("Survived", inplace=True, axis=1)
    df.drop("Pclass", inplace=True, axis=1)
    df.drop("Embarked", inplace=True, axis=1)

    return df


def init_tabular_llm(config: RailsConfig):
    """Initialize the model for searching tabular data."""
    # We just compute the titanic raw data frame
    titanic_csv_path = config.custom_data.get("tabular_data_path")
    raw_data_frame = _get_titanic_raw_data_frame(titanic_csv_path)

    model_config = _get_model_config(config, "tabular")
    model_path = model_config.parameters.get("path")

    # We just need to provide an empty data frame when initializing the model.
    empty_data_frame = pd.DataFrame()
    gpt = GPT4Pandas(model_path, empty_data_frame, verbose=False)

    tabular_llm = TabularLLM(gpt=gpt, raw_data_path=titanic_csv_path, raw_data_frame=raw_data_frame)

    register_llm_provider("tabular", get_llm_instance_wrapper(tabular_llm, "tabular"))


vectordb = None


def init_vectordb_model(config: RailsConfig):
    global vectordb
    model_config = _get_model_config(config, "vectordb")
    vectordb = _get_vector_db(
        model_name=model_config.model,
        data_path=config.custom_data["kb_data_path"],
        persist_path=model_config.parameters.get("persist_path"),
    )

    register_llm_provider("faiss", vectordb)


@action(is_system_action=True)
async def retrieve_relevant_chunks(
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
    tabular_llm: Optional[BaseLLM] = None,
):
    """Retrieve relevant chunks from the knowledge base and add them to the context."""
    user_message = context.get("last_user_message")

    # TODO: do this better using a separate canonical form
    if "csv" in user_message:
        llm_output = await tabular_llm.agenerate(prompts=[user_message])
        result, source_ref, citing_text = llm_output.generations[0][0].text.split("###")
    else:
        # using faiss vector database , pip install faiss-gpu if you have gpu, otherwise please use faiss-cpu
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        out = qa_chain(user_message)
        result = out["result"]
        citing_text = out["source_documents"][0].page_content
        source_ref = str(out["source_documents"][0].metadata["source"])

    context_updates = {
        "relevant_chunks": f"""
            Question: {user_message}
            Answer: {result},
            Citing : {citing_text},
            Source : {source_ref}
    """
    }

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


def init(llm_rails: LLMRails):
    config = llm_rails.config

    # Initialize the various models
    init_main_llm(config)
    init_vectordb_model(config)
    init_tabular_llm(config)

    # Register the custom `retrieve_relevant_chunks` for custom retrieval
    llm_rails.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
