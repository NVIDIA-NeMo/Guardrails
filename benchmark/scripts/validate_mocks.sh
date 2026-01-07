#!/usr/bin/env bash

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

#
# A script to check the health and model IDs of local OpenAI-compatible endpoints.
# Requires: curl, jq
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

TIMEOUT=3
SUMMARIES=()
ALL_PASSED=true

log_info() {
    echo -e "$1"
}

log_error() {
    echo -e "${RED}$1${NC}" >&2
}

log_warning() {
    echo -e "${YELLOW}$1${NC}"
}

log_success() {
    echo -e "${GREEN}$1${NC}"
}

# Check if required commands are available
check_dependencies() {
    for cmd in curl jq; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Error: '$cmd' is required but not installed."
            exit 1
        fi
    done
}

# Check an OpenAI-compatible endpoint for health and model availability
# Arguments: port, expected_model
check_endpoint() {
    local port=$1
    local expected_model=$2
    local base_url="http://localhost:${port}"
    local all_ok=true

    log_info "\n--- Checking Port: ${port} ---"

    # --- 1. Health Check ---
    local health_url="${base_url}/health"
    log_info "Checking ${health_url} ..."

    local response
    local http_code

    # Capture curl exit code to distinguish between connection error and timeout
    local curl_exit_code
    response=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" "$health_url" 2>/dev/null) || curl_exit_code=$?

    if [[ -n "${curl_exit_code:-}" ]]; then
        if [[ "$curl_exit_code" -eq 28 ]]; then
            log_error "Health Check FAILED: Connection timed out for port ${port}."
            log_error "--- Port ${port}: CHECKS FAILED ---"
            SUMMARIES+=("Port ${port} (${expected_model}): FAILED (Connection Timeout)")
        else
            log_error "Health Check FAILED: No response from server on port ${port}."
            log_error "--- Port ${port}: CHECKS FAILED ---"
            SUMMARIES+=("Port ${port} (${expected_model}): FAILED (Connection Error)")
        fi
        ALL_PASSED=false
        return 1
    fi

    http_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | sed '$d')

    if [[ "$http_code" != "200" ]]; then
        log_error "Health Check FAILED: Status code ${http_code}"
        all_ok=false
    else
        local status
        if status=$(echo "$body" | jq -r '.status' 2>/dev/null); then
            if [[ "$status" == "healthy" ]]; then
                log_success "Health Check PASSED: Status is 'healthy'."
            else
                log_warning "Health Check FAILED: Expected 'healthy', got '${status}'."
                all_ok=false
            fi
        else
            log_error "Health Check FAILED: Could not decode JSON response."
            all_ok=false
        fi
    fi

    # --- 2. Model Check ---
    local models_url="${base_url}/v1/models"
    log_info "Checking ${models_url} for '${expected_model}'..."

    # Capture curl exit code to distinguish between connection error and timeout
    curl_exit_code=""
    response=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" "$models_url" 2>/dev/null) || curl_exit_code=$?

    if [[ -n "${curl_exit_code:-}" ]]; then
        if [[ "$curl_exit_code" -eq 28 ]]; then
            log_error "Model Check FAILED: Connection timed out for port ${port}."
        else
            log_error "Model Check FAILED: No response from server on port ${port}."
        fi
        all_ok=false
    else
        http_code=$(echo "$response" | tail -n1)
        body=$(echo "$response" | sed '$d')

        if [[ "$http_code" != "200" ]]; then
            log_error "Model Check FAILED: Status code ${http_code}"
            all_ok=false
        else
            local model_ids
            if model_ids=$(echo "$body" | jq -r '.data[].id' 2>/dev/null); then
                if echo "$model_ids" | grep -qx "$expected_model"; then
                    log_success "Model Check PASSED: Found '${expected_model}' in model list."
                else
                    log_warning "Model Check FAILED: Expected '${expected_model}', but it was NOT found."
                    log_warning "Available models:"
                    echo "$model_ids" | while read -r model_id; do
                        log_warning "  - ${model_id}"
                    done
                    all_ok=false
                fi
            else
                log_error "Model Check FAILED: Could not decode JSON response."
                all_ok=false
            fi
        fi
    fi

    # --- Final Status ---
    if [[ "$all_ok" == true ]]; then
        log_success "--- Port ${port}: ALL CHECKS PASSED ---"
        SUMMARIES+=("Port ${port} (${expected_model}): PASSED")
        return 0
    else
        log_error "--- Port ${port}: CHECKS FAILED ---"
        SUMMARIES+=("Port ${port} (${expected_model}): FAILED")
        ALL_PASSED=false
        return 1
    fi
}

# Check the Rails configuration endpoint
# Arguments: port
check_rails_endpoint() {
    local port=$1
    local base_url="http://localhost:${port}"
    local endpoint="${base_url}/v1/rails/configs"
    local all_ok=true

    log_info "\n--- Checking Port: ${port} (Rails Config) ---"
    log_info "Checking ${endpoint} ..."

    local response
    local http_code
    local curl_exit_code=""

    # Capture curl exit code to distinguish between connection error and timeout
    response=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" "$endpoint" 2>/dev/null) || curl_exit_code=$?

    if [[ -n "${curl_exit_code:-}" ]]; then
        if [[ "$curl_exit_code" -eq 28 ]]; then
            log_error "Rails Check FAILED: Connection timed out for port ${port}."
        else
            log_error "Rails Check FAILED: No response from server on port ${port}."
        fi
        all_ok=false
    else
        http_code=$(echo "$response" | tail -n1)
        local body
        body=$(echo "$response" | sed '$d')

        # --- 1. HTTP Status Check ---
        if [[ "$http_code" == "200" ]]; then
            log_success "HTTP Status PASSED: Got ${http_code}."
        else
            log_warning "HTTP Status FAILED: Expected 200, got '${http_code}'."
            all_ok=false
        fi

        # --- 2. Body Content Check ---
        local is_array
        local array_len

        if is_array=$(echo "$body" | jq 'if type == "array" then true else false end' 2>/dev/null); then
            if [[ "$is_array" == "true" ]]; then
                array_len=$(echo "$body" | jq 'length' 2>/dev/null)
                if [[ "$array_len" -gt 0 ]]; then
                    log_success "Body Check PASSED: Response is an array with at least one entry."
                else
                    log_warning "Body Check FAILED: Response is not an array or is empty."
                    all_ok=false
                fi
            else
                log_warning "Body Check FAILED: Response is not an array or is empty."
                all_ok=false
            fi
        else
            log_error "Body Check FAILED: Could not decode JSON response."
            all_ok=false
        fi
    fi

    # --- Final Status ---
    if [[ "$all_ok" == true ]]; then
        log_success "--- Port ${port}: ALL CHECKS PASSED ---"
        SUMMARIES+=("Port ${port} (Rails Config): PASSED")
        return 0
    else
        log_error "--- Port ${port}: CHECKS FAILED ---"
        SUMMARIES+=("Port ${port} (Rails Config): FAILED")
        ALL_PASSED=false
        return 1
    fi
}

main() {
    log_info "Starting LLM endpoint health check..."

    check_dependencies

    # Run all checks (allow individual failures without exiting)
    check_endpoint 8000 "meta/llama-3.3-70b-instruct" || true
    check_endpoint 8001 "nvidia/llama-3.1-nemoguard-8b-content-safety" || true
    check_rails_endpoint 9000 || true

    log_info "\n--- Final Summary ---"

    for summary in "${SUMMARIES[@]}"; do
        log_info "$summary"
    done

    log_info "---------------------"

    if [[ "$ALL_PASSED" == true ]]; then
        log_success "Overall Status: All endpoints are healthy!"
        exit 0
    else
        log_error "Overall Status: One or more checks FAILED."
        exit 1
    fi
}

main "$@"
