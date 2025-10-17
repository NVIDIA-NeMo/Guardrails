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

from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.actions.actions import action


@action(name="summarize_document")
class SummarizeDocument:
    """Action for summarizing a document.

    This class provides a sample implementation of document summarization using LangChain's summarization chain.

    Args:
        document_path (str): The path to the document to be summarized.
        llm (BaseLLM): The Language Model for the summarization process.

    Example:
        ```python
        summarizer = SummarizeDocument(document_path="path/to/document.txt", llm=my_language_model)
        result = summarizer.run()
        print(result)  # The summarized document
        ```
    """

    def __init__(self, document_path: str, llm: BaseLLM):
        self.llm = llm
        self.document_path = document_path

    def run(self):
        summary_chain = load_summarize_chain(self.llm, "map_reduce")
        summarize_document_chain = AnalyzeDocumentChain(
            combine_docs_chain=summary_chain
        )
        try:
            with open(self.document_path) as f:
                document = f.read()
            summary = summarize_document_chain.run(document)
            return summary
        except Exception as e:
            print(f"Ran into an error while summarizing the document: {e}")
            return None
