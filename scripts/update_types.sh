#!/usr/bin/env bash

# Make sure Judgment backend server is running on port 8000
# This openapi_transform.py will get the relevant parts of the openapi.json file and save it to openapi.json
uv run scripts/openapi_transform.py > .openapi.json

# Generate the judgment_types.py file based on the schema in openapi.json.
datamodel-codegen \
  --input .openapi.json \
  --output src/judgeval/data/judgment_types.py \
  --output-model-type pydantic_v2.BaseModel \
  --target-python-version 3.10 \
  --use-annotated \
  --field-constraints \
  --use-default-kwarg \
  --formatters ruff-format \


# Generate the api_types.py file based on the schema in openapi.json.
datamodel-codegen \
  --input .openapi.json \
  --output src/judgeval/api/api_types.py \
  --output-model-type typing.TypedDict \
  --target-python-version 3.10 \
  --use-annotated \
  --use-default-kwarg \
  --formatters ruff-format \

# Generate the api.py file based on the schema in openapi.json.
uv run scripts/api_generator.py > src/judgeval/api/__init__.py

# Remove the openapi.json file since it is no longer needed
rm .openapi.json