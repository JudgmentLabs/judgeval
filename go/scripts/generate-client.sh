#!/bin/bash

python3 scripts/generate-client.py "${1:-http://localhost:8000/openapi.json}"
python3 scripts/generate-client-v1.py "${1:-http://localhost:8000/openapi.json}"
