#!/bin/bash

python3 scripts/generate_client.py "${1:-http://localhost:8000/openapi.json}"
