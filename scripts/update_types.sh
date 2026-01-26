#!/usr/bin/env bash

# Make sure judgeval server is running on port 10001
# Generate the v1 internal api files directly from the OpenAPI JSON endpoint
# The api_generator_v1.py script will handle fetching the JSON and generating both
# api_types.py and __init__.py

uv run scripts/api_generator_v1.py http://localhost:10001/openapi/json > src/judgeval/v1/internal/api/__init__.py