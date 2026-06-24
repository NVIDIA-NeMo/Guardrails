# Malicious dependency installs

Source: https://example.org/advisories/2026-dependency-confusion

Agents exposing an `install_package` tool have been steered into pulling known
typosquatted or malicious packages from a public index. The mitigation is a
denylist on the package name argument: the call is blocked when `name` is one of
the known-bad values, independent of the requested version.

<!-- params: {"arg_name": "name", "denied": ["leftpad-evil", "reqwest-utils"]} -->
