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

import csv
import json
import logging
from dataclasses import dataclass, field
from random import shuffle
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class Intent:
    """An intent tag having optional domain and canonical_form attributes"""

    intent_name: str
    domain: Optional[str] = None
    canonical_form: Optional[str] = None


@dataclass
class IntentExample:
    """A turn labeled with a specific intent tag/name"""

    SPLIT_TRAIN = "train"
    SPLIT_TEST = "test"
    SPLIT_VAL = "val"
    SPLIT_FULL = "full"

    intent: Intent
    text: str

    # Dataset split values are usually "train", "test" or "val"; however some datasets have other splits as well
    dataset_split: Optional[str] = None


@dataclass
class DatasetConnector:
    """A wrapper class to extract NLU specific data from a conversation dataset.

    In its current form, it can be used to extract intent samples and build
    the corresponding `user.co` Colang file.

    Attributes:
        name (str): The name of the dataset connector.
        intents (Set[Intent]): A set of unique Intent objects.
        slot_names (Set[str]): A set of unique slot names.
        domain_names (Set[str]): A set of unique domain names.
        intent_examples (List[IntentExample]): A list of IntentExample objects.

    Methods:
        read_dataset(dataset_path: str) -> None:
            Reads the dataset from the specified path, instantiating some or all of the fields of the object.
            E.g., it can instantiate intent names, slot names, intent examples, etc.

        get_intent_sample(intent_name: str, num_samples: int = 10) -> List[str]:
            Generates a random sample of `num_samples` texts for the `intent_name`.
            Inefficient implementation for now, as it passes through all intent samples to get the random subset.

        write_colang_output(output_file_name: str = None, num_samples_per_intent: int = 20) -> None:
            Creates an output file with pairs of turns and canonical forms.

    """

    name: str
    intents: Set[Intent] = field(default_factory=set)
    slot_names: Set[str] = field(default_factory=set)
    domain_names: Set[str] = field(default_factory=set)
    intent_examples: List[IntentExample] = field(default_factory=list)

    def read_dataset(self, dataset_path: str) -> None:
        """Reads the dataset from the specified path, instantiating some or all of the fields of the object.
        E.g. can instantiate intent names, slot names, intent examples etc.

        Args:
            dataset_path (str): The path to the conversation dataset.
        """
        raise NotImplementedError

    def get_intent_sample(self, intent_name: str, num_samples: int = 10) -> List[str]:
        """Generates a random sample of `num_samples` texts for the `intent_name`.
        Inefficient implementation for now, as it passes through all intent samples to get the random subset.

        Args:
            intent_name (str): The name of the intent.
            num_samples (int): The number of samples to generate.

        Returns:
            List[str]: A list of random samples for the specified intent.
        """
        all_samples_intent_name = []
        for intent in self.intent_examples:
            if intent.intent.intent_name == intent_name:
                all_samples_intent_name.append(intent.text)

        shuffle(all_samples_intent_name)
        if num_samples > 0:
            all_samples_intent_name = all_samples_intent_name[:num_samples]

        return all_samples_intent_name

    def write_colang_output(self, output_file_name: str = None, num_samples_per_intent: int = 20):
        """Creates an output file with pairs of turns and canonical forms.

        Args:
            output_file_name (str): The name of the output file.
            num_samples_per_intent (int): The number of samples per intent to include in the output.
        """
        if output_file_name is None:
            return

        sample_turns: Dict[str, List[str]] = dict()
        for intent in self.intents:
            if intent.canonical_form is None:
                print(f"Intent with no canonical form: {intent.intent_name} !")
                continue
            sample_intent_turns = self.get_intent_sample(
                intent_name=intent.intent_name, num_samples=num_samples_per_intent
            )
            sample_turns[intent.canonical_form] = sample_intent_turns

        for intent in self.intents:
            for intent2 in self.intents:
                if intent.canonical_form is None or intent2.canonical_form is None:
                    continue
                if intent.intent_name != intent2.intent_name and intent.canonical_form == intent2.canonical_form:
                    print(intent.intent_name + " -- " + intent2.intent_name)

        with open(output_file_name, "w", newline="\n") as output_file:
            for intent_canonical_form, intent_samples in sample_turns.items():
                output_file.write("define user " + intent_canonical_form + "\n")
                for intent_sample in intent_samples:
                    intent_sample = intent_sample.replace('"', "")
                    intent_sample = intent_sample.replace("\n", "")
                    output_file.write('  "' + intent_sample + '"\n')
                output_file.write("\n")


class Banking77Connector(DatasetConnector):
    BANKING77_FOLDER = "./banking/original_dataset/"
    BANKING77_CANONICAL_FORMS_FILE = "./banking/categories_canonical_forms.json"

    def __init__(self, name: str = "banking77"):
        super().__init__(name=name)

    @staticmethod
    def _read_canonical_forms(
        canonical_path: str = BANKING77_CANONICAL_FORMS_FILE,
    ) -> Dict[str, str]:
        """
        Reads the intent-canonical form mapping and returns it.

        Args:
            canonical_path (str): The path to the file containing the intent-canonical form mapping.

        Returns:
            dict: A dictionary mapping intent names to canonical forms.
        """
        intent_canonical_forms = dict()

        with open(canonical_path) as canonical_file:
            data = json.load(canonical_file)
            for intent_canonical_entry in data:
                if len(intent_canonical_entry) != 2:
                    print(
                        f"Problem: no canonical form found or too many canonical forms "
                        f"for entry {intent_canonical_entry}!"
                    )
                    continue
                intent = intent_canonical_entry[0]
                canonical_form = intent_canonical_entry[1]
                intent_canonical_forms[intent] = canonical_form

        return intent_canonical_forms

    def read_dataset(self, dataset_path: str = BANKING77_FOLDER) -> None:
        """
        Reads the dataset from the specified path, instantiating some or all of the fields of the object.
        E.g. can instantiate intent names, slot names, intent examples etc.

        Args:
            dataset_path (str): The path to the Banking77 dataset.
        """
        train_path = dataset_path + "train.csv"
        test_path = dataset_path + "test.csv"
        path_dict = {
            IntentExample.SPLIT_TRAIN: train_path,
            IntentExample.SPLIT_TEST: test_path,
        }

        intent_canonical_forms = Banking77Connector._read_canonical_forms()

        for dataset_type, dataset_path in path_dict.items():
            with open(dataset_path, "r") as banking_file:
                intent_examples = csv.reader(banking_file)

                for intent_example in intent_examples:
                    text = intent_example[0]
                    intent_name = intent_example[1]

                    # skip header if needed
                    if text == "text" and intent_name == "category":
                        continue

                    intent_canonical = None
                    if intent_name in intent_canonical_forms:
                        intent_canonical = intent_canonical_forms[intent_name]

                    intent = Intent(intent_name=intent_name, canonical_form=intent_canonical)
                    self.intents.add(intent)
                    self.intent_examples.append(IntentExample(intent=intent, text=text, dataset_split=dataset_type))


class ChitChatConnector(DatasetConnector):
    """A connector for handling ChitChat datasets.

    This class extends the DatasetConnector to read ChitChat datasets, including intent canonical forms.
    """

    CHITCHAT_FOLDER = "./chitchat/original_dataset/"
    CHITCHAT_CANONICAL_FORMS_FILE = "./chitchat/intent_canonical_forms.json"

    def __init__(self, name: str = "chitchat"):
        """Initialize a ChitChatConnector instance.

        Args:
            name (str): The name of the connector.
        """
        super().__init__(name=name)

    @staticmethod
    def _read_canonical_forms(
        canonical_path: str = CHITCHAT_CANONICAL_FORMS_FILE,
    ) -> Dict[str, str]:
        """Reads the intent-canonical form mapping and returns it.

        Args:
            canonical_path (str): The path to the intent-canonical forms file.

        Returns:
            Dict[str, str]: A dictionary mapping intent names to canonical forms.
        """
        intent_canonical_forms = dict()

        with open(canonical_path) as canonical_file:
            data = json.load(canonical_file)
            for intent_canonical_entry in data:
                if len(intent_canonical_entry) != 2:
                    print(
                        f"Problem: no canonical form found or too many canonical forms "
                        f"for entry {intent_canonical_entry}!"
                    )
                    continue
                intent = intent_canonical_entry[0]
                canonical_form = intent_canonical_entry[1]
                intent_canonical_forms[intent] = canonical_form

        return intent_canonical_forms

    def read_dataset(self, dataset_path: str = CHITCHAT_FOLDER) -> None:
        """Reads the ChitChat dataset from the specified path.

        Instantiates intent names, slot names, and intent examples.

        Args:
            dataset_path (str): The path to the ChitChat dataset.
        """
        full_dataset_path = dataset_path + "nlu.md"
        path_dict = {
            IntentExample.SPLIT_FULL: full_dataset_path,
        }

        intent_canonical_forms = ChitChatConnector._read_canonical_forms()

        intent_name = None
        intent_canonical = None
        intent = None
        for dataset_type, dataset_path in path_dict.items():
            with open(dataset_path, "r") as banking_file:
                # Read the markdown file in the Rasa markdown format
                lines = banking_file.readlines()

                for line in lines:
                    if line.startswith("##"):
                        intent_name = line[2:]
                        intent_start = "intent:"
                        pos = intent_name.find(intent_start)
                        if pos > 0:
                            intent_name = line[pos + len(intent_start) + 2 :]
                            intent_name = intent_name.strip()
                            intent_canonical = intent_canonical_forms.get(intent_name, None)

                            intent = Intent(intent_name=intent_name, canonical_form=intent_canonical)
                            self.intents.add(intent)

                    if line.startswith("- "):
                        text = line[2:]
                        text = text.strip()
                        if intent:
                            self.intent_examples.append(
                                IntentExample(intent=intent, text=text, dataset_split=dataset_type)
                            )
