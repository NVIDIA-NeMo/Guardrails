# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GLiNER FastAPI server for PII detection and entity extraction."""

import argparse
import logging
import os
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from gliner import GLiNER

from .models import (
    GLiNERRequest,
    GLiNERResponse,
    ModelInfo,
    ModelsResponse,
)
from .pii_utils import (
    DEFAULT_CATEGORIES,
    DEFAULT_LABELS,
    adjust_entity_positions,
    create_text_chunks,
    process_raw_entities,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment variables or defaults
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "1235"))
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/gliner-PII")
DEVICE = os.getenv("DEVICE", "auto")

# Initialize FastAPI app
app = FastAPI(
    title="GLiNER API",
    description=f"Running on {HOST}:{PORT} with chunking and deduplication",
    version="2.0.0",
)

# Global model variable
model = None


def extract_with_gliner(request: GLiNERRequest) -> GLiNERResponse:
    """
    GLiNER entity extraction with chunking, deduplication, and position tracking.

    Args:
        request: GLiNERRequest containing text and extraction parameters

    Returns:
        Dictionary with 'total_entities', 'entities' (list of EntitySpan), and 'tagged_text'
    """
    text = request.text
    labels = request.labels if request.labels is not None else DEFAULT_LABELS
    threshold = request.threshold
    chunk_length = request.chunk_length
    overlap = request.overlap
    flat_ner = request.flat_ner

    # Create all chunks with their offsets
    chunks, offsets = create_text_chunks(text, chunk_length, overlap)

    # Run inference on all chunks at once
    if model:
        batch_entities = model.inference(
            texts=chunks,
            labels=labels,
            threshold=threshold,
            flat_ner=flat_ner,
            relations=[],
        )

        if not (isinstance(batch_entities, list) and isinstance(batch_entities[0], list)):
            raise ValueError("Batch entities is expected to be a list of lists")
    else:
        raise ValueError("Model not loaded")

    entities = []
    for chunk_entities, offset in zip(batch_entities, offsets):
        adjusted = adjust_entity_positions(chunk_entities, offset)
        entities.extend(adjusted)

    # Process raw entities (deduplication, span conversion, tagging)
    return process_raw_entities(entities, text)


@app.on_event("startup")
async def load_model():
    """Load the GLiNER model on startup"""
    global model
    logger.info("Loading GLiNER model: %s", MODEL_NAME)
    logger.info("Server will run on: %s:%s", HOST, PORT)

    # Determine device
    if DEVICE == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = DEVICE

    try:
        model = GLiNER.from_pretrained(MODEL_NAME, map_location=device)
        logger.info("Model loaded successfully on %s", device)
        logger.info("API endpoint: http://%s:%s/v1", HOST, PORT)
        logger.info("Default labels: %d PII categories", len(DEFAULT_LABELS))
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """OpenAI-compatible models endpoint"""
    return ModelsResponse(data=[ModelInfo(id="gliner-ner", created=int(time.time()), owned_by="gliner")])


@app.post("/v1/extract", response_model=GLiNERResponse)
def extract_entities_advanced(request: GLiNERRequest):
    """Direct GLiNER endpoint with advanced processing"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return extract_with_gliner(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


@app.get("/v1/labels")
async def get_default_labels():
    """Get the default PII labels"""
    return {
        "labels": DEFAULT_LABELS,
        "count": len(DEFAULT_LABELS),
        "categories": DEFAULT_CATEGORIES,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with model stats"""
    device = "unknown"
    if model is not None:
        try:
            # Device reporting
            device = str(getattr(getattr(model, "model", model), "device", "unknown"))
        except Exception:
            device = "unknown"

    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_name": MODEL_NAME,
        "device": device,
        "server": f"{HOST}:{PORT}",
        "default_labels_count": len(DEFAULT_LABELS),
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GLiNER API with Chunking & Deduplication",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "server": f"{HOST}:{PORT}",
        "base_url": f"http://{HOST}:{PORT}/v1",
        "features": [
            "OpenAI-compatible API",
            "Text chunking with overlap",
            "Entity deduplication",
            "Comprehensive PII detection",
            "Configurable parameters",
        ],
        "documentation": f"http://{HOST}:{PORT}/docs",
    }


def main():
    """CLI entry point for the GLiNER server."""
    global HOST, PORT, MODEL_NAME, DEVICE

    parser = argparse.ArgumentParser(description="GLiNER API Server")
    parser.add_argument("--host", default=HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind to")
    parser.add_argument("--model", default=MODEL_NAME, help="GLiNER model to load")
    parser.add_argument(
        "--device",
        default=DEVICE,
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to use",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    # Update global variables
    HOST = args.host
    PORT = args.port
    MODEL_NAME = args.model
    DEVICE = args.device

    logger.info("Starting GLiNER API server...")
    logger.info("Host: %s", HOST)
    logger.info("Port: %s", PORT)
    logger.info("Model: %s", MODEL_NAME)
    logger.info("Device: %s", DEVICE)
    logger.info("PII Labels: %d", len(DEFAULT_LABELS))
    logger.info("Endpoint: http://%s:%s/v1", HOST, PORT)

    uvicorn.run(
        "gliner_server.server:app" if args.reload else app,
        host=HOST,
        port=PORT,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
