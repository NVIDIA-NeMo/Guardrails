#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
A script to check the health and model IDs of local OpenAI-compatible endpoints.
Requires the 'requests' library: pip install requests
"""

import logging
import sys

import requests

# --- Logging Setup ---
# Configure basic logging to print info-level messages
logging.basicConfig(level=logging.INFO, format="%(message)s")


def check_endpoint(port: int, expected_model: str):
    """
    Checks the /health and /v1/models endpoints for a standard
    OpenAI-compatible server.
    Returns a tuple: (bool success, str summary)
    """
    base_url = f"http://localhost:{port}"
    all_ok = True

    logging.info(f"\n--- Checking Port: {port} ---")

    # --- 1. Health Check ---
    health_url = f"{base_url}/health"
    logging.info(f"Checking {health_url} ...")
    try:
        response = requests.get(health_url, timeout=3)

        if response.status_code != 200:
            logging.error(f"Health Check FAILED: Status code {response.status_code}")
            all_ok = False
        else:
            try:
                data = response.json()
                status = data.get("status")
                if status == "healthy":
                    logging.info("Health Check PASSED: Status is 'healthy'.")
                else:
                    logging.warning(
                        f"Health Check FAILED: Expected 'healthy', got '{status}'."
                    )
                    all_ok = False
            except requests.exceptions.JSONDecodeError:
                logging.error("Health Check FAILED: Could not decode JSON response.")
                all_ok = False

    except requests.exceptions.ConnectionError:
        logging.error(f"Health Check FAILED: No response from server on port {port}.")
        logging.error(f"--- Port {port}: CHECKS FAILED ---")
        return False, f"Port {port} ({expected_model}): FAILED (Connection Error)"
    except requests.exceptions.Timeout:
        logging.error(f"Health Check FAILED: Connection timed out for port {port}.")
        logging.error(f"--- Port {port}: CHECKS FAILED ---")
        return False, f"Port {port} ({expected_model}): FAILED (Connection Timeout)"

    # --- 2. Model Check ---
    models_url = f"{base_url}/v1/models"
    logging.info(f"Checking {models_url} for '{expected_model}'...")
    try:
        response = requests.get(models_url, timeout=3)

        if response.status_code != 200:
            logging.error(f"Model Check FAILED: Status code {response.status_code}")
            all_ok = False
        else:
            try:
                data = response.json()
                models = data.get("data", [])
                model_ids = [model.get("id") for model in models]

                if expected_model in model_ids:
                    logging.info(
                        f"Model Check PASSED: Found '{expected_model}' in model list."
                    )
                else:
                    logging.warning(
                        f"Model Check FAILED: Expected '{expected_model}', but it was NOT found."
                    )
                    logging.warning("Available models:")
                    for model_id in model_ids:
                        logging.warning(f"  - {model_id}")
                    all_ok = False
            except requests.exceptions.JSONDecodeError:
                logging.error("Model Check FAILED: Could not decode JSON response.")
                all_ok = False
            except AttributeError:
                logging.error(
                    f"Model Check FAILED: Unexpected JSON structure in response from {models_url}."
                )
                all_ok = False

    except requests.exceptions.ConnectionError:
        logging.error(f"Model Check FAILED: No response from server on port {port}.")
        all_ok = False
    except requests.exceptions.Timeout:
        logging.error(f"Model Check FAILED: Connection timed out for port {port}.")
        all_ok = False

    # --- Final Status ---
    if all_ok:
        logging.info(f"--- Port {port}: ALL CHECKS PASSED ---")
        return True, f"Port {port} ({expected_model}): PASSED"
    else:
        logging.error(f"--- Port {port}: CHECKS FAILED ---")
        return False, f"Port {port} ({expected_model}): FAILED"


def check_rails_endpoint(port: int):
    """
    Checks the /v1/rails/configs endpoint for a specific 200 status
    and a non-empty list response.
    Returns a tuple: (bool success, str summary)
    """
    base_url = f"http://localhost:{port}"
    endpoint = f"{base_url}/v1/rails/configs"
    all_ok = True

    logging.info(f"\n--- Checking Port: {port} (Rails Config) ---")
    logging.info(f"Checking {endpoint} ...")

    try:
        response = requests.get(endpoint, timeout=3)

        # --- 1. HTTP Status Check ---
        if response.status_code == 200:
            logging.info(f"HTTP Status PASSED: Got {response.status_code}.")
        else:
            logging.warning(
                f"HTTP Status FAILED: Expected 200, got '{response.status_code}'."
            )
            all_ok = False

        # --- 2. Body Content Check ---
        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                logging.info(
                    "Body Check PASSED: Response is an array with at least one entry."
                )
            else:
                logging.warning(
                    "Body Check FAILED: Response is not an array or is empty."
                )
                logging.debug(
                    f"Response body (first 200 chars): {str(response.text)[:200]}"
                )
                all_ok = False
        except requests.exceptions.JSONDecodeError:
            logging.error("Body Check FAILED: Could not decode JSON response.")
            logging.debug(
                f"Response body (first 200 chars): {str(response.text)[:200]}"
            )
            all_ok = False

    except requests.exceptions.ConnectionError:
        logging.error(f"Rails Check FAILED: No response from server on port {port}.")
        all_ok = False
    except requests.exceptions.Timeout:
        logging.error(f"Rails Check FAILED: Connection timed out for port {port}.")
        all_ok = False

    # --- Final Status ---
    if all_ok:
        logging.info(f"--- Port {port}: ALL CHECKS PASSED ---")
        return True, f"Port {port} (Rails Config): PASSED"
    else:
        logging.error(f"--- Port {port}: CHECKS FAILED ---")
        return False, f"Port {port} (Rails Config): FAILED"


def main():
    """Run all health checks."""
    logging.info("Starting LLM endpoint health check...")

    check_results = [
        check_endpoint(8000, "meta/llama-3.3-70b-instruct"),
        check_endpoint(8001, "nvidia/llama-3.1-nemoguard-8b-content-safety"),
        check_rails_endpoint(9000),
    ]

    logging.info("\n--- Final Summary ---")

    all_passed = True
    for success, summary in check_results:
        logging.info(summary)
        if not success:
            all_passed = False

    logging.info("---------------------")

    if all_passed:
        logging.info("Overall Status: All endpoints are healthy!")
        sys.exit(0)
    else:
        logging.error("Overall Status: One or more checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()  # pragma: no cover
