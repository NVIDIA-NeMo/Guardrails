# Vertex AI Example

This guardrails configuration is a basic example using the Vertex AI API, and it can be adapted as needed.

Note that to call Vertex AI APIs, you need to perform [some initial setup](../../../../docs/user-guides/advanced/vertexai-setup.md), and to use Vertex AI with NeMo Guardrails, you additionally need to install the following:

```
pip install "google-cloud-aiplatform>=1.38.0"
pip install langchain-google-vertexai==0.1.0
```

The `gemini-1.0-pro` and `text-bison` models have been evaluated for topical rails, and `gemini-1.0-pro` has also been evaluated as a self-checking model for hallucination and content moderation. Evaluation results can be found [here](../../../../docs/evaluation/README.md).

**Disclaimer**: The Vertex AI models have only been tested on basic use cases, e.g., greetings and recognizing specific questions. On more complex queries, these models may not work correctly. Thorough testing and optimizations are needed before considering a production deployment. Additionally, as of March 14, 2024, there is [an open issue](https://github.com/GoogleCloudPlatform/generative-ai/issues/344) with Vertex AI models that result in them throwing an error likely triggered by moderation inside Vertex AI. In our tests, we find that self-checking with `gemini-1.0-pro` for hallucination and content moderation triggers this error. Production use cases should be designed to account for the possibility of this and handle it appropriately.
