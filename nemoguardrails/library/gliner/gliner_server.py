# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from gliner import GLiNER
from pydantic import BaseModel


def create_tagged_text(text: str, entities: List[Dict[str, Any]], label_key="suggested_label") -> str:
    """
    Create tagged text from original text and entities with positions.

    Args:
        text: Original text
        entities: List of entity dictionaries with 'value', label_key, 'start_position', 'end_position' keys

    Returns:
        Tagged text with format: [entity_text](entity_label)
    """
    if not entities:
        return text

    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x["start_position"])

    tagged_text = ""
    position = 0

    for entity in sorted_entities:
        start = entity["start_position"]
        end = entity["end_position"]
        entity_text = entity["value"]
        entity_label = entity[label_key]

        # Skip if this entity starts before our current position (overlap)
        if start < position:
            continue

        # Add text before the entity
        tagged_text += text[position:start]

        # Add the tagged entity
        tagged_text += f"[{entity_text}]({entity_label})"

        # Update position
        position = end

    # Add remaining text
    tagged_text += text[position:]

    return tagged_text


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

# Comprehensive PII labels
DEFAULT_LABELS = [
    "occupation",
    "certificate_license_number",
    "first_name",
    "date_of_birth",
    "ssn",
    "medical_record_number",
    "password",
    "unique_id",
    "phone_number",
    "national_id",
    "swift_bic",
    "company_name",
    "country",
    "license_plate",
    "tax_id",
    "employee_id",
    "pin",
    "state",
    "email",
    "date_time",
    "api_key",
    "biometric_identifier",
    "credit_debit_card",
    "coordinate",
    "device_identifier",
    "city",
    "postcode",
    "bank_routing_number",
    "vehicle_identifier",
    "health_plan_beneficiary_number",
    "url",
    "ipv4",
    "last_name",
    "cvv",
    "customer_id",
    "date",
    "user_name",
    "street_address",
    "ipv6",
    "account_number",
    "time",
    "age",
    "fax_number",
    "county",
    "gender",
    "sexuality",
    "political_view",
    "race_ethnicity",
    "religious_belief",
    "language",
    "blood_type",
    "mac_address",
    "http_cookie",
    "employment_status",
    "education_level",
]


class ChatMessage(BaseModel):
    role: str
    content: str


class GLiNERRequest(BaseModel):
    text: str
    labels: Optional[List[str]] = None
    threshold: Optional[float] = (
        0.5  # TODO idea: lower this threshold, add more entities list that the LLM gets with scores,
    )
    chunk_length: Optional[int] = 384
    overlap: Optional[int] = 128
    flat_ner: Optional[bool] = False


class ChatCompletionRequest(BaseModel):
    model: str = "gliner-ner"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    # GLiNER-specific parameters
    entity_labels: Optional[List[str]] = None
    threshold: Optional[float] = 0.5
    chunk_length: Optional[int] = 384
    overlap: Optional[int] = 128
    flat_ner: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class EntitySpan(BaseModel):
    value: str
    suggested_label: str
    start_position: int  # inclusive - character index where entity starts
    end_position: int  # exclusive - character index where entity ends (Python slicing style)
    score: float


class GLiNERResponse(BaseModel):
    entities: List[EntitySpan]  # List of entity spans with positions
    total_entities: int  # Total count of entities found
    tagged_text: str  # Tagged text with [entity](label) format


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "gliner"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


def extract_with_gliner(
    text: str,
    labels: List[str] = None,
    threshold: float = 0.5,
    chunk_length: int = 384,
    overlap: int = 128,
    flat_ner: bool = False,
):
    """
    GLiNER entity extraction with chunking, deduplication, and position tracking
    Returns entities with start/end positions, text, labels, and scores
    """
    if labels is None:
        labels = DEFAULT_LABELS

    start = 0
    entities = []

    # Chunking with overlap
    while start < len(text):
        temp_entities = model.predict_entities(
            text[start : start + chunk_length],
            labels,
            threshold=threshold,
            flat_ner=flat_ner,
        )
        for idx in range(len(temp_entities)):
            temp_entities[idx]["start"] += start
            temp_entities[idx]["end"] += start
        entities.extend(temp_entities)
        start += chunk_length - overlap

    # Deduplication - remove entities that are subsets of others
    entities_to_delete = []
    for idx, ent in enumerate(entities):
        has_superset = any(
            [
                i != idx and i not in entities_to_delete and e["start"] <= ent["start"] and e["end"] >= ent["end"]
                for i, e in enumerate(entities)
            ]
        )
        if has_superset:
            entities_to_delete.append(idx)
    for idx in sorted(entities_to_delete, reverse=True):
        del entities[idx]

    # Create dedup_map for both grouped entities and entity spans
    dedup_map = {}
    for ent in entities:
        label = ent.get("label")
        if not label:
            continue
        text_val = ent.get("text", "")
        score_val = float(ent.get("score", 0.0))
        key = (label, text_val.strip().lower())

        if key not in dedup_map or score_val > dedup_map[key]["score"]:
            dedup_map[key] = {
                "value": text_val,
                "suggested_label": label,
                "start_position": int(ent.get("start", 0)),
                "end_position": int(ent.get("end", 0)),
                "score": round(score_val, 3),
            }

    # Convert to list of EntitySpan objects
    entity_spans = [EntitySpan(**ent) for ent in dedup_map.values()]

    # Create tagged text
    tagged_text = create_tagged_text(text, list(dedup_map.values()))

    return {
        "total_entities": len(entity_spans),
        "entities": entity_spans,
        "tagged_text": tagged_text,
    }


@app.on_event("startup")
async def load_model():
    """Load the GLiNER model on startup"""
    global model
    print(f"Loading GLiNER model: {MODEL_NAME}")
    print(f"Server will run on: {HOST}:{PORT}")

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
        print(f"Model loaded successfully on {device}")
        print(f"API endpoint: http://{HOST}:{PORT}/v1")
        print(f"Default labels: {len(DEFAULT_LABELS)} PII categories")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """OpenAI-compatible models endpoint"""
    return ModelsResponse(data=[ModelInfo(id="gliner-ner", created=int(time.time()), owned_by="gliner")])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with GLiNER processing"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        text = user_messages[-1].content

        result = extract_with_gliner(
            text=text,
            labels=request.entity_labels or DEFAULT_LABELS,
            threshold=request.threshold,
            chunk_length=request.chunk_length,
            overlap=request.overlap,
            flat_ner=request.flat_ner,
        )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Convert EntitySpan objects to dictionaries for JSON serialization
        serializable_result = {
            "total_entities": result["total_entities"],
            "entities": [span.model_dump() for span in result["entities"]],
            "tagged_text": result["tagged_text"],
        }

        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=json.dumps(serializable_result)),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=len(text.split()),
                completion_tokens=len(json.dumps(serializable_result).split()),
                total_tokens=len(text.split()) + len(json.dumps(serializable_result).split()),
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/v1/extract", response_model=GLiNERResponse)
async def extract_entities_advanced(request: GLiNERRequest):
    """Direct GLiNER endpoint with advanced processing"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = extract_with_gliner(
            text=request.text,
            labels=request.labels or DEFAULT_LABELS,
            threshold=request.threshold,
            chunk_length=request.chunk_length,
            overlap=request.overlap,
            flat_ner=request.flat_ner,
        )

        return GLiNERResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


@app.get("/v1/labels")
async def get_default_labels():
    """Get the default PII labels"""
    return {
        "labels": DEFAULT_LABELS,
        "count": len(DEFAULT_LABELS),
        "categories": {
            "personal_identifiers": ["first_name", "last_name", "ssn", "date_of_birth"],
            "contact_info": [
                "email",
                "phone_number",
                "street_address",
                "city",
                "state",
            ],
            "financial": [
                "credit_debit_card",
                "cvv",
                "bank_routing_number",
                "account_number",
            ],
            "technical": ["ipv4", "ipv6", "mac_address", "url", "api_key"],
            "sensitive_attributes": [
                "gender",
                "sexuality",
                "race_ethnicity",
                "religious_belief",
            ],
        },
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
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "device": device,
        "server": f"{HOST}:{PORT}",
        "default_labels_count": len(DEFAULT_LABELS),
        "features": {
            "chunking": True,
            "overlap_processing": True,
            "entity_deduplication": True,
            "pii_detection": True,
        },
        "endpoints": {
            "base_url": f"http://{HOST}:{PORT}/v1",
            "chat_completions": "/v1/chat/completions",
            "extract": "/v1/extract",
            "models": "/v1/models",
            "labels": "/v1/labels",
            "health": "/health",
        },
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
    # Declare globals first
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

    print("Starting GLiNER API server...")
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"PII Labels: {len(DEFAULT_LABELS)}")
    print(f"Endpoint: http://{HOST}:{PORT}/v1")

    uvicorn.run(
        "gliner_server:app" if args.reload else app,
        host=HOST,
        port=PORT,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
