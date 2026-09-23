 ## Security

NVIDIA is dedicated to the security and trust of our software products and services, including all source code repositories managed through our organization.

If you need to report a security issue, please use the appropriate contact points outlined below. **Please do not report security vulnerabilities through GitHub.**

## Reporting Potential Security Vulnerability in an NVIDIA Product

To report a potential security vulnerability in any NVIDIA product:
- Web: [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
- E-Mail: psirt@nvidia.com
    - We encourage you to use the following PGP key for secure email communication: [NVIDIA public PGP Key for communication](https://www.nvidia.com/en-us/security/pgp-key)
    - Please include the following information:
   	 - Product/Driver name and version/branch that contains the vulnerability
     - Type of vulnerability (code execution, denial of service, buffer overflow, etc.)
   	 - Instructions to reproduce the vulnerability
   	 - Proof-of-concept or exploit code
   	 - Potential impact of the vulnerability, including how an attacker could exploit the vulnerability

While NVIDIA currently does not have a bug bounty program, we do offer acknowledgement when an externally reported security issue is addressed under our coordinated vulnerability disclosure policy. Please visit our [Product Security Incident Response Team (PSIRT)](https://www.nvidia.com/en-us/security/psirt-policies/) policies page for more information.

## Out of Scope

NeMo Guardrails is designed to be deployed as a component behind an authenticating network layer (for example, an API gateway, reverse proxy, service mesh sidecar such as `kube-rbac-proxy`, or load balancer) that is expected to provide authentication, authorization, TLS termination, and rate limiting before traffic reaches the library. See [Horizontal Scaling and Caching](docs/configure-rails/caching/model-memory-cache.mdx) and the [Actions Server deployment considerations](docs/run-rails/using-fastapi-server/actions-server.mdx) for the documented deployment model.

Because of this, the following are considered intentional design decisions rather than security vulnerabilities, and should not be reported through the security channels above. If in doubt, or if you have found a way to defeat the documented deployment model itself, please still report it so we can confirm:

- **No built-in authentication or authorization on the Guardrails server (`nemoguardrails server`) or the optional Actions Server API endpoints.** Authentication/authorization is expected to be enforced by the API gateway, reverse proxy, or sidecar in front of the deployment, not by the library itself.
- **No built-in TLS/mTLS termination on the FastAPI server.** TLS termination and certificate management are the responsibility of the ingress/gateway/proxy fronting the deployment.
- **No built-in rate limiting or request throttling on server endpoints.** These controls are expected to be enforced by the API gateway or load balancer layer.
- **No network isolation between the Guardrails server and the optional Actions Server by default.** Operators are responsible for placing the Actions Server on a separate network segment and adding authentication between the two services, as called out in the Actions Server deployment guidance.
- **Any already-authenticated/authorized caller being able to invoke any Guardrails API endpoint or any registered action.** Once a caller has passed the deployer's authentication/authorization layer, NeMo Guardrails does not currently perform additional fine-grained, per-endpoint or per-action authorization. Reports that only demonstrate this, without a way to bypass or escalate beyond what the deployer's own access-control layer already grants, are treated as hardening suggestions rather than vulnerabilities. Please open a regular GitHub issue for such hardening/feature requests instead of a security report.

We continue to welcome defense-in-depth hardening proposals (for example, an optional in-process authentication layer) as regular feature requests.

## NVIDIA Product Security

For all security-related concerns, please visit NVIDIA's Product Security portal at https://www.nvidia.com/en-us/security
