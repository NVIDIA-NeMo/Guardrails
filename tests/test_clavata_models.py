# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio

import pytest
from pydantic import ValidationError

import nemoguardrails.library.clavata.request as request_module
from nemoguardrails.http import HTTPConnectionError, HTTPResponse, RetryingHTTPClient
from nemoguardrails.library.clavata.actions import LabelResult, PolicyResult
from nemoguardrails.library.clavata.errs import (
    ClavataPluginAPIError,
    ClavataPluginAPIRateLimitError,
    ClavataPluginValueError,
)
from nemoguardrails.library.clavata.request import (
    _CLAVATA_RETRY_POLICY,
    ClavataClient,
    CreateJobResponse,
    Job,
    Report,
    Result,
    SectionReport,
)
from nemoguardrails.testing import RecordingHTTPClient


@pytest.mark.unit
class TestLabelResult:
    def test_from_section_report_matched(self):
        """Test LabelResult creation from a SectionReport with a match"""
        section_report = SectionReport(name="TestLabel", message="Test message", result="OUTCOME_TRUE")

        label_result = LabelResult.from_section_report(section_report)

        assert label_result.label == "TestLabel"
        assert label_result.message == "Test message"
        assert label_result.matched is True

    def test_from_section_report_not_matched(self):
        """Test LabelResult creation from a SectionReport without a match"""
        section_report = SectionReport(name="TestLabel", message="Test message", result="OUTCOME_FALSE")

        label_result = LabelResult.from_section_report(section_report)

        assert label_result.label == "TestLabel"
        assert label_result.message == "Test message"
        assert label_result.matched is False

    def test_from_section_report_failed(self):
        """Test LabelResult creation from a SectionReport that failed"""
        section_report = SectionReport(name="TestLabel", message="Test message", result="OUTCOME_FAILED")

        label_result = LabelResult.from_section_report(section_report)

        assert label_result.label == "TestLabel"
        assert label_result.message == "Test message"
        assert label_result.matched is False


@pytest.mark.unit
class TestPolicyResult:
    def test_from_report_matched(self):
        """Test PolicyResult creation from a Report with matches"""
        report = Report(
            result="OUTCOME_TRUE",
            sectionEvaluationReports=[
                SectionReport(name="Label1", message="Message 1", result="OUTCOME_TRUE"),
                SectionReport(name="Label2", message="Message 2", result="OUTCOME_FALSE"),
            ],
        )

        policy_result = PolicyResult.from_report(report)

        assert policy_result.failed is False
        assert policy_result.policy_matched is True
        assert len(policy_result.label_matches) == 2

        assert policy_result.label_matches[0].label == "Label1"
        assert policy_result.label_matches[0].message == "Message 1"
        assert policy_result.label_matches[0].matched is True

        assert policy_result.label_matches[1].label == "Label2"
        assert policy_result.label_matches[1].message == "Message 2"
        assert policy_result.label_matches[1].matched is False

    def test_from_report_not_matched(self):
        """Test PolicyResult creation from a Report without matches"""
        report = Report(
            result="OUTCOME_FALSE",
            sectionEvaluationReports=[
                SectionReport(name="Label1", message="Message 1", result="OUTCOME_FALSE"),
                SectionReport(name="Label2", message="Message 2", result="OUTCOME_FALSE"),
            ],
        )

        policy_result = PolicyResult.from_report(report)

        assert policy_result.failed is False
        assert policy_result.policy_matched is False
        assert len(policy_result.label_matches) == 2
        assert all(not label.matched for label in policy_result.label_matches)

    def test_from_report_failed(self):
        """Test PolicyResult creation from a failed Report"""
        report = Report(result="OUTCOME_FAILED", sectionEvaluationReports=[])

        policy_result = PolicyResult.from_report(report)

        assert policy_result.failed is True
        assert policy_result.policy_matched is False
        assert len(policy_result.label_matches) == 0

    def test_from_job_completed_with_matches(self):
        """Test PolicyResult creation from a completed Job with matches"""
        job = Job(
            status="JOB_STATUS_COMPLETED",
            results=[
                Result(
                    report=Report(
                        result="OUTCOME_TRUE",
                        sectionEvaluationReports=[
                            SectionReport(
                                name="Label1",
                                message="Message 1",
                                result="OUTCOME_TRUE",
                            ),
                            SectionReport(
                                name="Label2",
                                message="Message 2",
                                result="OUTCOME_FALSE",
                            ),
                        ],
                    )
                )
            ],
        )

        policy_result = PolicyResult.from_job(job)

        assert policy_result.failed is False
        assert policy_result.policy_matched is True
        assert len(policy_result.label_matches) == 2
        assert policy_result.label_matches[0].matched is True
        assert policy_result.label_matches[1].matched is False

    def test_from_job_completed_without_matches(self):
        """Test PolicyResult creation from a completed Job without matches"""
        job = Job(
            status="JOB_STATUS_COMPLETED",
            results=[Result(report=Report(result="OUTCOME_FALSE", sectionEvaluationReports=[]))],
        )

        policy_result = PolicyResult.from_job(job)

        assert policy_result.failed is False
        assert policy_result.policy_matched is False
        assert len(policy_result.label_matches) == 0

    def test_from_job_failed(self):
        """Test PolicyResult creation from a failed Job"""
        job = Job(status="JOB_STATUS_FAILED", results=[])

        policy_result = PolicyResult.from_job(job)

        assert policy_result.failed is True
        assert policy_result.policy_matched is False
        assert len(policy_result.label_matches) == 0

    def test_from_job_canceled(self):
        """Test PolicyResult creation from a canceled Job"""
        job = Job(status="JOB_STATUS_CANCELED", results=[])

        policy_result = PolicyResult.from_job(job)

        assert policy_result.failed is True
        assert policy_result.policy_matched is False
        assert len(policy_result.label_matches) == 0

    def test_from_job_pending(self):
        """Test PolicyResult creation from a pending Job raises an error"""
        job = Job(status="JOB_STATUS_PENDING", results=[])

        with pytest.raises(ClavataPluginAPIError) as excinfo:
            PolicyResult.from_job(job)

        assert "Policy evaluation is not complete" in str(excinfo.value)

    def test_from_job_running(self):
        """Test PolicyResult creation from a running Job raises an error"""
        job = Job(status="JOB_STATUS_RUNNING", results=[])

        with pytest.raises(ClavataPluginAPIError) as excinfo:
            PolicyResult.from_job(job)

        assert "Policy evaluation is not complete" in str(excinfo.value)

    def test_from_job_invalid_result_count(self):
        """Test PolicyResult creation with multiple results raises an error"""
        job = Job(
            status="JOB_STATUS_COMPLETED",
            results=[
                Result(report=Report(result="OUTCOME_TRUE", sectionEvaluationReports=[])),
                Result(report=Report(result="OUTCOME_FALSE", sectionEvaluationReports=[])),
            ],
        )

        with pytest.raises(ClavataPluginAPIError) as excinfo:
            PolicyResult.from_job(job)

        assert "Expected 1 report per job" in str(excinfo.value)


def _job_response_json(sections, status="JOB_STATUS_COMPLETED", result="OUTCOME_TRUE"):
    return {
        "job": {
            "status": status,
            "results": [
                {
                    "report": {
                        "result": result,
                        "sectionEvaluationReports": [
                            {"name": name, "message": message, "result": section_result}
                            for name, message, section_result in sections
                        ],
                    }
                }
            ],
        }
    }


@pytest.mark.unit
class TestCreateJobResponse:
    def test_model_validate_valid_json(self):
        """Test that CreateJobResponse can parse a valid JSON response"""
        sections = [
            ("DogBarking", "Content contains references to dog barking", "OUTCOME_TRUE"),
            ("CatMeowing", "No cat sounds detected", "OUTCOME_FALSE"),
        ]

        response = CreateJobResponse.model_validate(_job_response_json(sections))

        assert response.job.status == "JOB_STATUS_COMPLETED"
        assert len(response.job.results) == 1
        report = response.job.results[0].report
        assert report.result == "OUTCOME_TRUE"
        assert [(s.name, s.message, s.result) for s in report.sectionEvaluationReports] == sections

    def test_model_validate_missing_fields(self):
        """Test that CreateJobResponse validation fails on missing required fields"""
        # Missing 'results' field
        json_data = {
            "job": {
                "status": "JOB_STATUS_COMPLETED"
                # Missing 'results' field
            }
        }

        with pytest.raises(ValidationError):
            CreateJobResponse.model_validate(json_data)

    def test_model_validate_invalid_enum_values(self):
        """Test that CreateJobResponse validation fails on invalid enum values"""
        # Invalid status
        json_data = {
            "job": {"status": "INVALID_STATUS", "results": []}  # Invalid status
        }

        with pytest.raises(ValidationError):
            CreateJobResponse.model_validate(json_data)

    def test_model_validate_complete_job_response(self):
        """Test that CreateJobResponse can parse a complete job response with nested structures"""
        sections = [
            ("Label1", "First section matched", "OUTCOME_TRUE"),
            ("Label2", "Second section did not match", "OUTCOME_FALSE"),
            ("Label3", "Third section evaluation failed", "OUTCOME_FAILED"),
        ]

        response = CreateJobResponse.model_validate(_job_response_json(sections))

        assert response.job.status == "JOB_STATUS_COMPLETED"
        assert len(response.job.results) == 1
        report = response.job.results[0].report
        assert report.result == "OUTCOME_TRUE"
        assert [(s.name, s.message, s.result) for s in report.sectionEvaluationReports] == sections


@pytest.fixture
def zero_sleep_clavata_retry(monkeypatch):
    monkeypatch.setattr(
        ClavataClient,
        "_retrying_http_client",
        staticmethod(
            lambda client: RetryingHTTPClient(
                client,
                _CLAVATA_RETRY_POLICY,
                sleep=lambda _: asyncio.sleep(0),
            )
        ),
    )


@pytest.mark.unit
class TestClavataClientTransport:
    @pytest.mark.asyncio
    async def test_clavata_client_uses_shared_retry_policy(self, zero_sleep_clavata_retry):
        response = CreateJobResponse(
            job=Job(
                status="JOB_STATUS_COMPLETED",
                results=[Result(report=Report(result="OUTCOME_FALSE", sectionEvaluationReports=[]))],
            )
        )
        transport = RecordingHTTPClient(
            [
                HTTPResponse(status_code=429),
                HTTPResponse(status_code=200, content=response.model_dump_json().encode()),
            ]
        )
        job = await ClavataClient(
            "https://clavata.example",
            api_key="test-key",
            http_client=transport,
        ).create_job("hello", "policy-id")

        assert job.status == "JOB_STATUS_COMPLETED"
        assert len(transport.requests) == 2
        request = transport.requests[1]
        assert request.method == "POST"
        assert request.url == "https://clavata.example/v1/jobs"
        assert request.headers == {"Authorization": "Bearer test-key"}
        assert request.json == {
            "content_data": [{"text": "hello"}],
            "policy_id": "policy-id",
            "wait_for_completion": True,
        }
        assert request.timeout is None

    @pytest.mark.asyncio
    async def test_clavata_client_closes_owned_shared_client(self, monkeypatch):
        response = CreateJobResponse(
            job=Job(
                status="JOB_STATUS_COMPLETED",
                results=[Result(report=Report(result="OUTCOME_FALSE", sectionEvaluationReports=[]))],
            )
        )
        transport = RecordingHTTPClient([HTTPResponse(status_code=200, content=response.model_dump_json().encode())])
        monkeypatch.setattr(request_module, "create_http_client", lambda: transport)

        job = await ClavataClient(
            "https://clavata.example",
            api_key="test-key",
        ).create_job("hello", "policy-id")

        assert job.status == "JOB_STATUS_COMPLETED"
        assert transport.close_calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("response", "error", "message"),
        [
            (
                HTTPResponse(status_code=503, content=b"unavailable"),
                ClavataPluginAPIError,
                "Clavata call failed with status code 503",
            ),
            (
                HTTPResponse(status_code=200, content=b"not-json"),
                ClavataPluginValueError,
                "Failed to parse Clavata response as JSON",
            ),
            (
                HTTPResponse(status_code=200, content=b'{"unexpected":true}'),
                ClavataPluginValueError,
                "Invalid response format from Clavata API",
            ),
        ],
    )
    async def test_clavata_client_preserves_response_failure_type(self, response, error, message):
        transport = RecordingHTTPClient([response])

        with pytest.raises(error, match=message):
            await ClavataClient(
                "https://clavata.example",
                api_key="test-key",
                http_client=transport,
            ).create_job("hello", "policy-id")

    @pytest.mark.asyncio
    async def test_clavata_client_does_not_retry_ambiguous_transport_failure(self):
        transport = RecordingHTTPClient([HTTPConnectionError("connection lost")])

        with pytest.raises(ClavataPluginAPIError):
            await ClavataClient(
                "https://clavata.example",
                api_key="test-key",
                http_client=transport,
            ).create_job("hello", "policy-id")

        assert len(transport.requests) == 1

    @pytest.mark.asyncio
    async def test_clavata_client_preserves_rate_limit_error(self, zero_sleep_clavata_retry):
        transport = RecordingHTTPClient(
            [
                HTTPResponse(status_code=429),
                HTTPResponse(status_code=429),
                HTTPResponse(status_code=429),
            ]
        )

        with pytest.raises(ClavataPluginAPIRateLimitError):
            await ClavataClient(
                "https://clavata.example",
                api_key="test-key",
                http_client=transport,
            ).create_job("hello", "policy-id")

        assert len(transport.requests) == 3
