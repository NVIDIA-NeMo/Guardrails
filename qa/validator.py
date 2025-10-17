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


def are_strings_semantically_same(string1, string2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer().fit_transform([string1, string2])
    similarity = cosine_similarity(vectorizer)
    # Determine if the similarity exceeds a threshold (e.g., 0.5)
    if similarity[0][1] > 0.5:
        return True
    else:
        return False


if __name__ == "__main__":
    # Example usage
    string1 = "Hello, how are you doing?"
    # string1 = "What is 434 + 56*7.5?"
    string2 = "Hi, how are you?"
    # string2 = "Goodbye, see you next time!"
    # string2 =  "Sorry, I cannot comment on anything which is not relevant to the jobs report."
    if are_strings_semantically_same(string1, string2):
        print("The strings are semantically the same.")
    else:
        print("The strings are not semantically the same.")
