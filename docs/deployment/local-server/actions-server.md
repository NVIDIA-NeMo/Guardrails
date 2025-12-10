# Actions Server

The Actions Server enables you to run the actions invoked from the guardrails more securely (see [Security Guidelines](../../security/guidelines.md) for more details). The action server should be deployed in a separate environment.

```{note}
Even though highly recommended for production deployments, using an *actions server* is optional and configured per guardrails configuration. If no actions server is specified in a guardrails configuration, the actions will run in the same process as the guardrails server.
```

To launch the server:

```sh
nemoguardrails actions-server [--port PORT]
```

On startup, the actions server will automatically register all predefined actions and all actions in the current folder (including sub-folders).

## Endpoints

The OpenAPI specification for the actions server is available at `http://localhost:8001/redoc` or `http://localhost:8001/docs`.

### `/v1/actions/list`

To list the [available actions](../../python-api/index.md#actions) for the server, use the `/v1/actions/list` endpoint.

```text
GET /v1/actions/list
```

Sample response:

```json
["apify","bing_search","google_search","google_serper","openweather_query","searx_search","serp_api_query","wikipedia_query","wolframalpha_query","zapier_nla_query"]
```

### `/v1/actions/run`

To execute an action with a set of parameters, use the `/v1/actions/run` endpoint:

```text
POST /v1/actions/run
```

```json
{
    "action_name": "wolfram_alpha_request",
    "action_parameters": {
      "query": "What is the largest prime factor for 1024?"
    }
}
```

Sample response:

```json
{
  "status": "success",
  "result": "2"
}
```
