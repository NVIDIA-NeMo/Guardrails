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

import pytest
from pydantic import ValidationError

from nemoguardrails.library.clavata.actions import LabelResult, PolicyResult
from nemoguardrails.library.clavata.errs import ClavataPluginAPIError
from nemoguardrails.library.clavata.request import (
    CreateJobResponse,
    Job,
    Report,
    Result,
    SectionReport,
)


@pytest.mark.unit
class TestLabelResult:
    def test_from_section_report_matched(self):
        """Test LabelResult creation from a SectionReport with a match"""
        section_report = SectionReport(
            name="TestLabel", message="Test message", result="OUTCOME_TRUE"
        )

        label_result = LabelResult.from_section_report(section_report)

        assert label_result.label == "TestLabel"
        assert label_result.message == "Test message"
        assert label_result.matched is True

    def test_from_section_report_not_matched(self):
        """Test LabelResult creation from a SectionReport without a match"""
        section_report = SectionReport(
            name="TestLabel", message="Test message", result="OUTCOME_FALSE"
        )

        label_result = LabelResult.from_section_report(section_report)

        assert label_result.label == "TestLabel"
        assert label_result.message == "Test message"
        assert label_result.matched is False

    def test_from_section_report_failed(self):
        """Test LabelResult creation from a SectionReport that failed"""
        section_report = SectionReport(
            name="TestLabel", message="Test message", result="OUTCOME_FAILED"
        )

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
                SectionReport(
                    name="Label1", message="Message 1", result="OUTCOME_TRUE"
                ),
                SectionReport(
                    name="Label2", message="Message 2", result="OUTCOME_FALSE"
                ),
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
                SectionReport(
                    name="Label1", message="Message 1", result="OUTCOME_FALSE"
                ),
                SectionReport(
                    name="Label2", message="Message 2", result="OUTCOME_FALSE"
                ),
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
            results=[
                Result(
                    report=Report(result="OUTCOME_FALSE", sectionEvaluationReports=[])
                )
            ],
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
                Result(
                    report=Report(result="OUTCOME_TRUE", sectionEvaluationReports=[])
                ),
                Result(
                    report=Report(result="OUTCOME_FALSE", sectionEvaluationReports=[])
                ),
            ],
        )

        with pytest.raises(ClavataPluginAPIError) as excinfo:
            PolicyResult.from_job(job)

        assert "Expected 1 report per job" in str(excinfo.value)


@pytest.mark.unit
class TestCreateJobResponse:
    def test_model_validate_valid_json(self):
        """Test that CreateJobResponse can parse a valid JSON response"""
        # This simulates the parsed JSON that would come from an API response
        json_data = {
            "job": {
                "status": "JOB_STATUS_COMPLETED",
                "results": [
                    {
                        "report": {
                            "result": "OUTCOME_TRUE",
                            "sectionEvaluationReports": [
                                {
                                    "name": "DogBarking",
                                    "message": "Content contains references to dog barking",
                                    "result": "OUTCOME_TRUE",
                                },
                                {
                                    "name": "CatMeowing",
                                    "message": "No cat sounds detected",
                                    "result": "OUTCOME_FALSE",
                                },
                            ],
                        }
                    }
                ],
            }
        }

        # This is what would happen in _make_request after resp.json()
        response = CreateJobResponse.model_validate(json_data)

        # Verify the response was parsed correctly
        assert response.job.status == "JOB_STATUS_COMPLETED"
        assert len(response.job.results) == 1
        assert response.job.results[0].report.result == "OUTCOME_TRUE"
        assert len(response.job.results[0].report.sectionEvaluationReports) == 2

        # Check first section
        section1 = response.job.results[0].report.sectionEvaluationReports[0]
        assert section1.name == "DogBarking"
        assert section1.message == "Content contains references to dog barking"
        assert section1.result == "OUTCOME_TRUE"

        # Check second section
        section2 = response.job.results[0].report.sectionEvaluationReports[1]
        assert section2.name == "CatMeowing"
        assert section2.message == "No cat sounds detected"
        assert section2.result == "OUTCOME_FALSE"

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
        json_data = {
            "job": {
                "status": "JOB_STATUS_COMPLETED",
                "results": [
                    {
                        "report": {
                            "result": "OUTCOME_TRUE",
                            "sectionEvaluationReports": [
                                {
                                    "name": "Label1",
                                    "message": "First section matched",
                                    "result": "OUTCOME_TRUE",
                                },
                                {
                                    "name": "Label2",
                                    "message": "Second section did not match",
                                    "result": "OUTCOME_FALSE",
                                },
                                {
                                    "name": "Label3",
                                    "message": "Third section evaluation failed",
                                    "result": "OUTCOME_FAILED",
                                },
                            ],
                        }
                    }
                ],
            }
        }

        response = CreateJobResponse.model_validate(json_data)

        # Check overall response
        assert response.job.status == "JOB_STATUS_COMPLETED"

        # Check results
        assert len(response.job.results) == 1
        report = response.job.results[0].report
        assert report.result == "OUTCOME_TRUE"

        # Check sections
        sections = report.sectionEvaluationReports
        assert len(sections) == 3

        # Verify all section fields
        assert sections[0].name == "Label1"
        assert sections[0].message == "First section matched"
        assert sections[0].result == "OUTCOME_TRUE"

        assert sections[1].name == "Label2"
        assert sections[1].message == "Second section did not match"
        assert sections[1].result == "OUTCOME_FALSE"

        assert sections[2].name == "Label3"
        assert sections[2].message == "Third section evaluation failed"
        assert sections[2].result == "OUTCOME_FAILED"
