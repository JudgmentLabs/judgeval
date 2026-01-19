#!/usr/bin/env bash

# Make sure judgeval server is running on port 100001
# This openapi_transform.py will get the openapi.json file and save it to openapi.json


uv run scripts/openapi_transform.py > .openapi.json

# Generate the v1 internal api files based on the schema in openapi.json.
uv run scripts/api_generator_v1.py .openapi.json > src/judgeval/v1/internal/api/__init__.py

# Remove the openapi.json file since it is no longer needed
rm .openapi.json