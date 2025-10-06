# Security Guidelines

Allowing LLMs to access external resources – such as search interfaces, databases, or computing resources such as Wolfram Alpha – can dramatically improve their capabilities. However, the unpredictable nature of LLM completion generations means that – without careful integration – these external resources can potentially be manipulated by attackers, leading to a dramatic increase in the risk of deployment of these combined models.

This document sets out guidelines and principles for providing LLMs access to external data and compute resources in a safe and secure way.

## The Golden Rule

Consider the LLM to be, in effect, a web browser under the complete control of the user, and all content it generates is untrusted. Any service that is invoked **must** be invoked in the context of the LLM user. When designing an internal API (see below) between a resource and an LLM, ask yourself *“Would I deliberately expose this resource with this interface directly to the internet?”*  If the answer is “no”, you should rethink your integration.

## Assumed Interaction Model

![](../_static/images/llm-api-interaction-model.png)

We assume that the data flow for accessing external resources has the following logical components:

1. The LLM, which receives a prompt as input and produces text as output.

2. A parsing/dispatch engine, which examines LLM output for an indication that a call to an external resource is needed. It is responsible for the following:
   - Identifying that one or more external resources must be called
   - Identifying the specific resources requested and extracting the parameters to be included in the external call
   - Calling the internal API associated with the requested resources with the correct parameters, including any authentication and/or authorization information associated with the LLM user
   - Receiving the responses
   - Re-introducing the responses into the LLM prompt in the correct location with the correct formatting, and returning it to the process managing the LLM for the next LLM execution
3. An internal API acting as a gateway between the parsing/dispatch engine and a single external resource. These APIs should have hard-coded URLs, endpoints, paths, etc., wherever possible, designed to minimize attack surfaces. It is responsible for the following:
   - Verifying that the user currently authenticated to the LLM is authorized to call the requested external resource with the requested parameters
   - Validating the input
   - Interacting with the external resource and receiving a response, including any authentication
   - Validating the response
   - Returning the response to the dispatch engine

The parsing step may take on a number of forms, including pre-loading the LLM with tokens or verbs to indicate specific actions, or doing some form of embedding search on lines of the output. It is currently common practice to include a specific verb (e.g., “FINISH”) to indicate that the LLM should return the result to the user – effectively making user interaction an external resource as well – however, this area is new enough that there is no such thing as a “standard practice”.

We separate the internal APIs from the parsing/dispatch engine for the following reasons:
1. Keeping validation and authorization code co-located with the relevant API or service
2. Keeping any authentication information required for the external API isolated from the LLM (to prevent leaks)
3. Enabling more modular development of external resources for LLM use, and reducing the impact of external API changes.

## Specific Guidelines

### Fail gracefully and secretly - do not disclose details of services

When a resource cannot be accessed for any reason, including due to a malformed request or inadequate authorization, the internal API should return a message that the LLM can respond to appropriately. Error messages from the external API should be trapped and rewritten. The text response to the parsing engine should not indicate what external API was called or why it failed. The parsing engine should be responsible for taking failures due to lack of authorization and reconstructing the LLM generation as though the attempt to call the resource did not happen, and taking other non-authorization-related failures and returning a nonspecific failure message that does not reveal specifics of the integration.

It should be assumed that users of the service will attempt to discover internal APIs and/or verbs that their specific prompt or LLM session does not enable and that they do not have the authorization to use; a user should not be able to detect that some internal API exists based on interactions with the LLM.

### Log all interactions

At a minimum, the following should be recorded:

1. Text that triggered an action from the parsing/dispatch engine
2. How that text was parsed to an internal API call, and what the parameters were
3. Authorization information provided to the internal API (including: method and time of authn/authz, expiration or duration of same, scope/role information, user name or UUID, etc.)
4. What call was made from the internal API to the external API, as well as the result
5. How the resulting text was re-inserted into the LLM prompt

### Track user authorization and security scope to external resources

If authorization is required to access the LLM, the corresponding authorization information should be provided to the resource; all calls to that resource should execute in the authorization context of the user. If a user is not authorized to access a resource, attempts to use that resource should fail.

For instance, accessing a company database must only be done when the user interacting with the LLM is themselves authorized to access those records in that database. Allowing execution of code within a python session should only be allowed when the user attempting to induce the LLM to do so would be permitted to execute arbitrary commands on the service that runs the interpreter.

### Parameterize and validate all inputs and outputs

Any requests to external services should be parameterized and have strict validation requirements. These parameters should be injected into audited templates matched against validated versions of the external APIs with user control restricted to the minimum set of viable parameters. Particular care should be paid to potential code injection routes (e.g., SQL injection; injection of comment characters for python; open redirects in search queries, etc.) and risk of remote file (or data) inclusion in responses. To the extent possible, values returned from external APIs should also be validated against expected contents and formats to prevent injection or unintended behaviors.

In addition to validation requirements, as above, all outputs should be examined for private information before being returned to the parsing/dispatch engine, particularly leaked API keys, user information, API information, etc. APIs reflecting information such as user authentication, IP addresses, the context in which the LLM is accessing a resource, etc., may all be anticipated to be a persistent headache that must be proactively designed against.

### Avoid persisting changes when possible

Requests from the LLM to the external API should avoid producing a persistent change of state unless required for the functionality of the service. Performing high-risk actions such as: creating or dropping a table; downloading a file; writing an arbitrary file to disk; establishing and nohupping a process; should all be explicitly disallowed unless specifically required. In such cases, the internal API should be associated with an internal service role that isolates the ability to make and persist these changes. Where possible, consider other usage patterns that will allow the same effect to be achieved without requiring LLM external services to perform them directly (e.g., providing a link to a pre-filled form for scheduling an appointment which a user could modify before submitting).

### Any persistent changes should be made via a parameterized interface

When the main functionality of the external API is to record some persistent state (e.g., scheduling an appointment), those updates should be entirely parameterized and strongly validated. Any information recorded by such an API should be tied to the requesting user, and the ability of any user to retrieve that information, either for themselves or any other user, should be carefully evaluated and controlled.

### Prefer allow-lists and fail-closed

Wherever possible, any external interface should default to denying requests, with specific permitted requests and actions placed on an allow list.

### Isolate all authentication information from the LLM

The LLM should have no ability to access any authentication information for external resources; any keys, passwords, security tokens, etc., should only be accessible to the internal API service that calls the external resource. The calling service should also be responsible for verifying the authorization of the user to access the resource in question, either by internal authorization checks or by interacting with the external service. As noted above, all information regarding any errors, authorization failures, etc., should be removed from the text output and returned to the parsing service.

### Engage with security teams proactively to assess interfaces

Integrating LLMs with external resources is inherently an exercise in API security. When designing these interfaces, early and timely involvement with security experts can reduce the risk associated with these interfaces as well as speed development.

## Adversarial testing

AI safety and security is a community effort, and this is one of the main reasons we have released NeMo Guardrails to the community. We hope to bring many developers and enthusiasts together to build better solutions for Trustworthy AI. Our initial release is a starting point. We have built a collection of guardrails and educational examples that provide helpful controls and resist a variety of common attacks, however, they are not perfect. We have conducted adversarial testing on these example bots and will soon release a whitepaper on a larger-scale study. Here are some items to watch out for when creating your own bots:

1. Over-aggressive moderation: Some of the AI Safety rails, can occasionally block otherwise safe requests. This is more likely to happen when multiple guardrails are used together. One possible strategy to resolve this is to use logic in the flow to reduce unnecessary calls; for example to call fact-checking only for factual questions.
2. Overgeneralization of canonical forms: NeMo Guardrails uses canonical forms like `ask about jobs report` to guide its behavior and to generalize to situations not explicitly defined in the Colang configuration. It may occasionally get the generalization wrong, so that guardrails miss certain examples or trigger unexpectedly. If this happens, it can often be improved by adding or adjusting the `define user` forms in the [Colang files](../user-guides/colang-language-syntax-guide.md), or modifying the sample conversations in the [configuration](../user-guides/configuration-guide.md).
3. Nondeterminism: LLMs use a concept known as *temperature*, as well as other techniques, to introduce variation in their responses. This creates a much more natural experience, however, it can on occasion create unexpected behavior in LLM applications that can be difficult to reproduce. As with all AI applications, it is a good practice to use thorough evaluation and regression-testing suites.

## Conclusion

Integrating external resources into LLMs can dramatically improve their capabilities and make them significantly more valuable to end users. However, any increase in expressive power comes with an increase in potential risk. To avoid potentially catastrophic risks, including unauthorized information disclosure all the way up to remote code execution, the interfaces that allow LLMs to access these external resources must be carefully and thoughtfully designed from a security-first perspective.
