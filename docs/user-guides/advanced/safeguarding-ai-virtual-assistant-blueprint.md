# Blueprint with NemoGuard NIMs

```{note}
This documentation is intended for developers that want to integrate NeMo Guardrails into [NVIDIA Blueprints](https://build.nvidia.com/blueprints)
```

NVIDIA Blueprints are comprehensive reference workflows that accelerate AI application development and deployment. They make it easy to start building and setting up virtual assistants, offering ready-made workflows and tools. Whether you need a simple AI-powered chatbot or a fully animated digital human interface, NVIDIA provides resources to help you create an AI assistant that’s scalable and aligned with your brand. For example, developers can use the **[NVIDIA AI Blueprint for AI virtual assistants](https://build.nvidia.com/nvidia/ai-virtual-assistant-for-customer-service)** to build an AI assistant for customer service for delivering a responsive, efficient customer support experience.

## Prerequisites and Setup

### NVIDIA Blueprint

The NVIDIA Blueprint for AI virtual assistants can be deployed either with the NVIDIA-hosted endpoints or with locally hosted NIM microservices. To get started with the deployment:

- **[Using the NVIDIA-hosted endpoint](https://build.nvidia.com/nvidia/ai-virtual-assistant-for-customer-service)**: The launchable opens to a jupyter notebook - `ai-virtual-assitant-notebook.ipynb`. Make sure that the [compute requirements](https://github.com/NVIDIA-AI-Blueprints/ai-virtual-assistant?tab=readme-ov-file#hardware-requirements) are satisfied

### Integrating NemoGuard

First create a `nemoguardrails.yaml` file at `ai-virtual-assistant/deploy/compose` directory to download and initialize NeMo Guardrails toolkit. Make sure it has the following content in it. You can change the port as per your requirement.

```yaml
services:
  nemoguardrails:
    build:
      context: ../../../nemoguardrails  # Path to the cloned repository
      dockerfile: Dockerfile
    container_name: nemoguardrails
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      NGC_API_KEY: ${NGC_API_KEY}  # NVIDIA GPU Cloud API Key
    ports:
      - "8650:8650"  # Map container port 8000 to host port 8000
      - "8000:8000"
    user: "${USER_ID:-1000}:${GROUP_ID:-1000}"  # Use the current user's UID/GID
    command: >
      bash -c "
      export OPENAI_API_KEY=${OPENAI_API_KEY} &&
      cd nemoguardrails
      pip install --no-cache-dir -e .[all]
      "

```

Follow the notebook from - `nemoguardrails/examples/notebooks/safeguard_ai_virtual_assistant_notebook.ipynb` to deploy the NemoGuard NIM using downloadable containers and learn to step by step integrate it into the NIM blueprint

### NIM Blueprint API Usage

Use the `ai-virtual-assistant/notebooks/api_usage.ipynb` to run the blueprint api to see the agent response
