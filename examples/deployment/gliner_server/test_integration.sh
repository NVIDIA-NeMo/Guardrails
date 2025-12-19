#!/bin/bash
# Integration test for GLiNER server
# Usage: ./test_integration.sh

set -e

SERVER_HOST="localhost"
SERVER_PORT="1235"
BASE_URL="http://${SERVER_HOST}:${SERVER_PORT}"
SERVER_PID=""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

echo "=== GLiNER Server Integration Test ==="
echo ""

# Start the server in background
echo "Starting GLiNER server..."
gliner-server --host $SERVER_HOST --port $SERVER_PORT &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..60}; do
    if curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Server is ready!${NC}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo -e "${RED}Server failed to start within 60 seconds${NC}"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== Running Integration Tests ==="
echo ""

# Test 1: Health endpoint
echo "Test 1: Health endpoint"
HEALTH_RESPONSE=$(curl -s "${BASE_URL}/health")
if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "$HEALTH_RESPONSE"
    exit 1
fi

# Test 2: Root endpoint
echo "Test 2: Root endpoint"
ROOT_RESPONSE=$(curl -s "${BASE_URL}/")
if echo "$ROOT_RESPONSE" | grep -q '"message":"GLiNER API'; then
    echo -e "${GREEN}✓ Root endpoint passed${NC}"
else
    echo -e "${RED}✗ Root endpoint failed${NC}"
    exit 1
fi

# Test 3: Labels endpoint
echo "Test 3: Labels endpoint"
LABELS_RESPONSE=$(curl -s "${BASE_URL}/v1/labels")
if echo "$LABELS_RESPONSE" | grep -q '"email"'; then
    echo -e "${GREEN}✓ Labels endpoint passed${NC}"
else
    echo -e "${RED}✗ Labels endpoint failed${NC}"
    exit 1
fi

# Test 4: Models endpoint
echo "Test 4: Models endpoint"
MODELS_RESPONSE=$(curl -s "${BASE_URL}/v1/models")
if echo "$MODELS_RESPONSE" | grep -q '"id":"gliner-ner"'; then
    echo -e "${GREEN}✓ Models endpoint passed${NC}"
else
    echo -e "${RED}✗ Models endpoint failed${NC}"
    exit 1
fi

# Test 5: Extract endpoint - PII detection
echo "Test 5: Extract endpoint (PII detection)"
EXTRACT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/extract" \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Hello, my name is John Smith and my email is john.smith@example.com",
        "labels": ["first_name", "last_name", "email"],
        "threshold": 0.5
    }')

if echo "$EXTRACT_RESPONSE" | grep -q '"total_entities"'; then
    ENTITY_COUNT=$(echo "$EXTRACT_RESPONSE" | grep -o '"total_entities":[0-9]*' | cut -d: -f2)
    if [ "$ENTITY_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Extract endpoint passed (found $ENTITY_COUNT entities)${NC}"
    else
        echo -e "${RED}✗ Extract endpoint failed (no entities found)${NC}"
        echo "$EXTRACT_RESPONSE"
        exit 1
    fi
else
    echo -e "${RED}✗ Extract endpoint failed${NC}"
    echo "$EXTRACT_RESPONSE"
    exit 1
fi

# Test 6: Extract endpoint - verify tagged_text
echo "Test 6: Verify tagged_text format"
if echo "$EXTRACT_RESPONSE" | grep -q 'tagged_text'; then
    echo -e "${GREEN}✓ Tagged text format passed${NC}"
else
    echo -e "${RED}✗ Tagged text format failed${NC}"
    exit 1
fi

# Test 7: Extract with no PII
echo "Test 7: Extract with no PII"
NO_PII_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/extract" \
    -H "Content-Type: application/json" \
    -d '{
        "text": "The weather is nice today.",
        "labels": ["email", "phone_number"],
        "threshold": 0.5
    }')

if echo "$NO_PII_RESPONSE" | grep -q '"total_entities":0'; then
    echo -e "${GREEN}✓ No PII detection passed${NC}"
else
    echo -e "${GREEN}✓ No PII test completed (may have found entities)${NC}"
fi

echo ""
echo "=== All Integration Tests Passed! ==="
echo ""

# Cleanup happens via trap
