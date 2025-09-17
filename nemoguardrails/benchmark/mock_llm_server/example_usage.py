#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Example usage of the Mock LLM Server.

This script demonstrates how to interact with the running mock server
using standard HTTP requests and the OpenAI Python client.
"""

import json
import time

import requests


def test_with_requests():
    """Test the server using the requests library."""
    base_url = "http://localhost:8000"

    print("Testing Mock LLM Server with requests library...")
    print("=" * 50)

    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health check: {response.status_code} - {response.json()}")
    except requests.RequestException as e:
        print(f"Health check failed: {e}")
        print("Make sure the server is running: python run_server.py")
        return

    # Test models endpoint
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        print(f"\\nModels: {response.status_code}")
        models_data = response.json()
        for model in models_data["data"]:
            print(f"  - {model['id']}")
    except requests.RequestException as e:
        print(f"Models request failed: {e}")

    # Test chat completion
    try:
        chat_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50,
        }
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=chat_payload,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        print(f"\\nChat completion: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data['choices'][0]['message']['content']}")
            print(f"Usage: {data['usage']}")
    except requests.RequestException as e:
        print(f"Chat completion failed: {e}")

    # Test text completion
    try:
        completion_payload = {
            "model": "text-davinci-003",
            "prompt": "The capital of France is",
            "max_tokens": 10,
        }
        response = requests.post(
            f"{base_url}/v1/completions",
            json=completion_payload,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        print(f"\\nText completion: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data['choices'][0]['text']}")
            print(f"Usage: {data['usage']}")
    except requests.RequestException as e:
        print(f"Text completion failed: {e}")


def test_with_openai_client():
    """Test the server using the OpenAI Python client."""
    try:
        import openai
    except ImportError:
        print("\\nOpenAI client not available. Install with: pip install openai")
        return

    print("\\n" + "=" * 50)
    print("Testing with OpenAI client library...")
    print("=" * 50)

    # Configure client to use local server
    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy-key",  # Server doesn't validate, but client requires it
    )

    try:
        # Test chat completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello from OpenAI client!"}],
        )
        print(f"Chat completion response: {response.choices[0].message.content}")
        print(
            f"Usage: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}"
        )

        # Test text completion (if supported by client version)
        try:
            response = client.completions.create(
                model="text-davinci-003", prompt="OpenAI client test: ", max_tokens=10
            )
            print(f"Text completion response: {response.choices[0].text}")
        except Exception as e:
            print(f"Text completion not supported in this OpenAI client version: {e}")

    except Exception as e:
        print(f"OpenAI client test failed: {e}")


def benchmark_performance():
    """Simple performance benchmark."""
    print("\\n" + "=" * 50)
    print("Performance Benchmark")
    print("=" * 50)

    base_url = "http://localhost:8000"
    num_requests = 10

    chat_payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Benchmark test"}],
        "max_tokens": 20,
    }

    print(f"Making {num_requests} chat completion requests...")

    start_time = time.time()
    successful_requests = 0

    for i in range(num_requests):
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=chat_payload,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if response.status_code == 200:
                successful_requests += 1
        except requests.RequestException:
            pass

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Results:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful requests: {successful_requests}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per request: {total_time/num_requests:.3f} seconds")
    print(f"  Requests per second: {num_requests/total_time:.2f}")


def main():
    """Main function to run all tests."""
    print("Mock LLM Server Example Usage")
    print("=" * 50)
    print("Make sure the server is running before running this script:")
    print("  python run_server.py")
    print()

    # Test with requests
    test_with_requests()

    # Test with OpenAI client
    test_with_openai_client()

    # Simple benchmark
    benchmark_performance()

    print("\\nExample completed!")


if __name__ == "__main__":
    main()
