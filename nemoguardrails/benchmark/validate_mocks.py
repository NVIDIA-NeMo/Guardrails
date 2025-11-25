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
Requires the 'httpx' library: pip install httpx
"""

import json
import logging
import sys

import httpx

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

    logging.info("\n--- Checking Port: %s ---", port)

    # --- 1. Health Check ---
    health_url = f"{base_url}/health"
    logging.info("Checking %s ...", health_url)
    try:
        response = httpx.get(health_url, timeout=3)

        if response.status_code != 200:
            logging.error("Health Check FAILED: Status code %s", response.status_code)
            all_ok = False
        else:
            try:
                data = response.json()
                status = data.get("status")
                if status == "healthy":
                    logging.info("Health Check PASSED: Status is 'healthy'.")
                else:
                    logging.warning("Health Check FAILED: Expected 'healthy', got '%s'.", status)
                    all_ok = False
            except json.JSONDecodeError:
                logging.error("Health Check FAILED: Could not decode JSON response.")
                all_ok = False

    except httpx.ConnectError:
        logging.error("Health Check FAILED: No response from server on port %s.", port)
        logging.error("--- Port %s: CHECKS FAILED ---", port)
        return False, "Port %s (%s): FAILED (Connection Error)" % (port, expected_model)
    except httpx.TimeoutException:
        logging.error("Health Check FAILED: Connection timed out for port %s.", port)
        logging.error("--- Port %s: CHECKS FAILED ---", port)
        return False, "Port %s (%s): FAILED (Connection Timeout)" % (
            port,
            expected_model,
        )

    # --- 2. Model Check ---
    models_url = f"{base_url}/v1/models"
    logging.info("Checking %s for '%s'...", models_url, expected_model)
    try:
        response = httpx.get(models_url, timeout=3)

        if response.status_code != 200:
            logging.error("Model Check FAILED: Status code %s", response.status_code)
            all_ok = False
        else:
            try:
                data = response.json()
                models = data.get("data", [])
                model_ids = [model.get("id") for model in models]

                if expected_model in model_ids:
                    logging.info("Model Check PASSED: Found '%s' in model list.", expected_model)
                else:
                    logging.warning(
                        "Model Check FAILED: Expected '%s', but it was NOT found.",
                        expected_model,
                    )
                    logging.warning("Available models:")
                    for model_id in model_ids:
                        logging.warning("  - %s", model_id)
                    all_ok = False
            except json.JSONDecodeError:
                logging.error("Model Check FAILED: Could not decode JSON response.")
                all_ok = False
            except AttributeError:
                logging.error(
                    "Model Check FAILED: Unexpected JSON structure in response from %s.",
                    models_url,
                )
                all_ok = False

    except httpx.ConnectError:
        logging.error("Model Check FAILED: No response from server on port %s.", port)
        all_ok = False
    except httpx.TimeoutException:
        logging.error("Model Check FAILED: Connection timed out for port %s.", port)
        all_ok = False

    # --- Final Status ---
    if all_ok:
        logging.info("--- Port %s: ALL CHECKS PASSED ---", port)
        return True, "Port %s (%s): PASSED" % (port, expected_model)
    else:
        logging.error("--- Port %s: CHECKS FAILED ---", port)
        return False, "Port %s (%s): FAILED" % (port, expected_model)


def check_rails_endpoint(port: int):
    """
    Checks the /v1/rails/configs endpoint for a specific 200 status
    and a non-empty list response.
    Returns a tuple: (bool success, str summary)
    """
    base_url = f"http://localhost:{port}"
    endpoint = f"{base_url}/v1/rails/configs"
    all_ok = True

    logging.info("\n--- Checking Port: %s (Rails Config) ---", port)
    logging.info("Checking %s ...", endpoint)

    try:
        response = httpx.get(endpoint, timeout=3)

        # --- 1. HTTP Status Check ---
        if response.status_code == 200:
            logging.info("HTTP Status PASSED: Got %s.", response.status_code)
        else:
            logging.warning("HTTP Status FAILED: Expected 200, got '%s'.", response.status_code)
            all_ok = False

        # --- 2. Body Content Check ---
        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                logging.info("Body Check PASSED: Response is an array with at least one entry.")
            else:
                logging.warning("Body Check FAILED: Response is not an array or is empty.")
                logging.debug("Response body (first 200 chars): %s", str(response.text)[:200])
                all_ok = False
        except json.JSONDecodeError:
            logging.error("Body Check FAILED: Could not decode JSON response.")
            logging.debug("Response body (first 200 chars): %s", str(response.text)[:200])
            all_ok = False

    except httpx.ConnectError:
        logging.error("Rails Check FAILED: No response from server on port %s.", port)
        all_ok = False
    except httpx.TimeoutException:
        logging.error("Rails Check FAILED: Connection timed out for port %s.", port)
        all_ok = False

    # --- Final Status ---
    if all_ok:
        logging.info("--- Port %s: ALL CHECKS PASSED ---", port)
        return True, "Port %s (Rails Config): PASSED" % port
    else:
        logging.error("--- Port %s: CHECKS FAILED ---", port)
        return False, "Port %s (Rails Config): FAILED" % port


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
