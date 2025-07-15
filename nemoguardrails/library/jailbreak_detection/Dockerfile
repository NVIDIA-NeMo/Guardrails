# Use python:3.10 as the base image
FROM python:3.10-slim

# Install git and wget
RUN apt-get update && apt-get install -y git gcc g++ python3-dev wget && apt-get clean

# Predownload embedding-based jailbreak detection models, set environment variable for path
WORKDIR /models
RUN wget https://huggingface.co/nvidia/NemoGuard-JailbreakDetect/resolve/main/snowflake.pkl
ENV EMBEDDING_CLASSIFIER_PATH=/models

# Set working directory
WORKDIR /app

# Copy only requirements.txt
COPY requirements.txt .

# Upgrade pip and install the minimal set of requirements for jailbreak detection Server
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# Set the device on which the model should load e.g., "cpu", "cuda:0", etc.
ENV JAILBREAK_CHECK_DEVICE=cpu

# Predownload the GPT2 model.
RUN python -c "from transformers import GPT2LMHeadModel, GPT2TokenizerFast; GPT2LMHeadModel.from_pretrained('gpt2-large'); GPT2TokenizerFast.from_pretrained('gpt2-large');"

# Expose a port for the server
EXPOSE 1337

# Start the server as the default command
ENTRYPOINT ["/usr/local/bin/python", "/app/server.py"]
CMD ["--port=1337"]
