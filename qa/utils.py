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

import select
import time
from unittest import TestCase

from .chatter import close_chatter, create_chatter
from .logger import create_logger
from .validator import are_strings_semantically_same

TIMEOUT = 15  # in seconds


class ExampleConfigChatterTestCase(TestCase):
    """Helper TestCase for testing an example configuration."""

    logger = None
    chatter = None
    example_name = None
    config_name = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Create a logger and a chatter
        cls.logger = create_logger(f"{cls.example_name}.log")
        cls.chatter = create_chatter(cls.example_name, cls.config_name, cls.logger)

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        close_chatter(cls.chatter)

    def run_test(self, messages):
        """Test the jailbreak_check example"""
        self.logger.info(f"Running {self.example_name} ...")

        if self.chatter is not None:
            # Process the questions and validate the answers
            for question, expected_answers in messages.items():
                self.logger.info(f"User: {question}")

                # Send the question to chatter
                self.chatter.stdin.write(question + "\n")
                self.chatter.stdin.flush()

                # Read the answer from chatter
                start_time = time.time()
                while True:
                    # Check if the process has completed
                    retcode = self.chatter.poll()
                    if retcode is not None:
                        break

                    rlist, _, _ = select.select([self.chatter.stdout], [], [], 0.1)
                    if rlist:
                        output = self.chatter.stdout.readline().strip()
                        if output:
                            self.logger.info(f"output: {output}")
                            break

                    if time.time() - start_time > TIMEOUT:
                        self.logger.error(
                            "Timeout reached. No non-empty line received."
                        )
                        break

                # Validate the answer
                if len([answer for answer in expected_answers if answer in output]) > 0:
                    assert True
                elif (
                    len(
                        [
                            answer
                            for answer in expected_answers
                            if are_strings_semantically_same(answer, output)
                        ]
                    )
                    > 0
                ):
                    assert True
                else:
                    assert (
                        False
                    ), f"The output '{output}' is NOT expected as the bot's response."
