# Vertex AI Setup

This guide outlines how to get set up Vertex AI enabling calling of Vertex AI APIs from code.

In order to use Vertex AI, you need to perform some initial setup with the Google Cloud Platform (GCP).

1. Create a GCP account: The following [page](https://cloud.google.com/docs/get-started) provides more information about the Google Cloud Platform and how to get started. In your account [create a project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) and [set up billing for it](https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_an_existing_project)
2. Install the `gcloud` CLI ([guide](https://cloud.google.com/sdk/docs/install)). Note that although 3.8 - 3.12 are listed as supported, [this error](https://stackoverflow.com/questions/77316716/gcloud-modulenotfounderror-no-module-named-imp) occurs on Python 3.12. This guide was tested using Python 3.10.2.
3. Create a service account following [this guide](https://cloud.google.com/iam/docs/service-accounts-create) and grant it the role of `Vertex AI Service Agent`.
4. Create and download a service account key for the service account ([guide](https://cloud.google.com/iam/docs/keys-create-delete)).
5. Enable the Vertex AI API ([guide](https://cloud.google.com/vertex-ai/docs/start/cloud-environment#:~:text=Enable%20Vertex%20AI%20APIs,-In%20the%20Google&text=Click%20Enable%20All%20Recommended%20APIs,the%20APIs%20are%20being%20enabled.))
6. Install additional python libraries needed to call Vertex AI using `pip install "google-cloud-aiplatform>=1.38.0"`

Test that you are successfully able to call VertexAI APIs using the following snippet:

```python
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "<path>/<to>/<your>/<service>/<account>/<key>.json"

from vertexai.preview.generative_models import GenerativeModel, ChatSession

model = GenerativeModel("gemini-1.0-pro")
chat = model.start_chat()

def get_chat_response(chat: ChatSession, prompt: str):
    response = chat.send_message(prompt)
    return response.text

prompts = [
    "Hi, who are you?",
    "What can you tell me about the United States?",
    "Where was its 44th president born?",
]

for prompt in prompts:
    print("User:", prompt)
    print("Gemini:", get_chat_response(chat, prompt))
    print("------")
```
