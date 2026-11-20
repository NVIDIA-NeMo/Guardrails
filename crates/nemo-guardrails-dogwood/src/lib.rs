// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A thin Python shell over `dogwood-ops`, the operations the `dogwood` CLI and Dogwood's own language bindings share.
//! It adds no semantics: every parse, lowering, type check and decision is Dogwood's. Values cross as JSON text, in
//! the shapes `dogwood-ops` defines (`EventInput` in, `AuthorizerDecision` and `ReplayReport` out).

use dogwood_ops::{EventInput, LiveAuthorizer, OpError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A Dogwood error as a Python `ValueError` whose message is the error's JSON report.
fn error(e: OpError) -> PyErr {
    PyValueError::new_err(serde_json::to_string(&e).unwrap_or_else(|_| format!("{e:?}")))
}

/// Lower a policy set and replay a `.log` trace through a fresh authorizer, as the bindings' `replay` does (with the
/// fail-closed rule): the JSON `ReplayReport`.
#[pyfunction]
#[pyo3(signature = (source, action_schema, log, event_schema=None, providers=None, macros=None))]
fn replay(
    source: &str,
    action_schema: &str,
    log: &str,
    event_schema: Option<&str>,
    providers: Option<&str>,
    macros: Option<&str>,
) -> PyResult<String> {
    let inputs = dogwood_ops::inputs(event_schema, providers, macros).map_err(error)?;
    let report = dogwood_ops::replay(source, &inputs, action_schema, log).map_err(error)?;
    serde_json::to_string(&report).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Lower a policy set against its schemas: the JSON `LowerArtifacts` (the lowered Cedar policies and the augmented
/// schema, text and Cedar JSON). A parse, macro or lowering failure raises `ValueError`.
#[pyfunction]
#[pyo3(signature = (source, action_schema, event_schema=None, providers=None, macros=None))]
fn lower(
    source: &str,
    action_schema: &str,
    event_schema: Option<&str>,
    providers: Option<&str>,
    macros: Option<&str>,
) -> PyResult<String> {
    let inputs = dogwood_ops::inputs(event_schema, providers, macros).map_err(error)?;
    let artifacts =
        dogwood_ops::ops::lower_to_cedar(source, &inputs, action_schema).map_err(error)?;
    serde_json::to_string(&artifacts).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Parse, lower and type-check a policy set: the JSON `ValidateReport` (`passed`, `errors`, `warnings`).
#[pyfunction]
#[pyo3(signature = (source, action_schema, event_schema=None, providers=None, macros=None))]
fn validate(
    source: &str,
    action_schema: &str,
    event_schema: Option<&str>,
    providers: Option<&str>,
    macros: Option<&str>,
) -> PyResult<String> {
    let inputs = dogwood_ops::inputs(event_schema, providers, macros).map_err(error)?;
    let report =
        dogwood_ops::ops::validate_policies(source, &inputs, action_schema).map_err(error)?;
    serde_json::to_string(&report).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Dogwood's live authorizer: one instance is one ordered history. Events are fed with their own timestamps (seconds),
/// so a history kept elsewhere can be replayed into it.
#[pyclass(unsendable, name = "Authorizer")]
struct Authorizer {
    inner: LiveAuthorizer,
}

#[pymethods]
impl Authorizer {
    #[new]
    #[pyo3(signature = (source, action_schema, event_schema=None, providers=None, macros=None))]
    fn new(
        source: &str,
        action_schema: &str,
        event_schema: Option<String>,
        providers: Option<String>,
        macros: Option<String>,
    ) -> PyResult<Self> {
        let inner = LiveAuthorizer::new(source, action_schema, event_schema, providers, macros)
            .map_err(error)?;
        Ok(Self { inner })
    }

    /// Submit one event (JSON `EventInput`). A decision event returns the JSON `AuthorizerDecision`; a history event
    /// returns None. An event Dogwood rejects raises `ValueError`.
    fn submit(&mut self, event: &str) -> PyResult<Option<String>> {
        let event: EventInput =
            serde_json::from_str(event).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let decision = self.inner.is_authorized(&event).map_err(error)?;
        decision
            .map(|d| serde_json::to_string(&d).map_err(|e| PyValueError::new_err(e.to_string())))
            .transpose()
    }

    /// Start a fresh history with the same lowered policies.
    fn reset(&mut self) -> PyResult<()> {
        self.inner.reset().map_err(error)
    }
}

#[pymodule]
fn nemoguardrails_dogwood(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(replay, m)?)?;
    m.add_function(wrap_pyfunction!(lower, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_class::<Authorizer>()?;
    Ok(())
}
