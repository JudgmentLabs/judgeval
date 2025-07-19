#!/bin/zsh

# Make sure Judgment backend server is running on port 8000
# This openapi_transform.py will get the relevant parts of the openapi.json file and save it to openapi_new.json
uv run judgeval/data/scripts/openapi_transform.py > judgeval/data/openapi_new.json

# Then, datamodel-codegen will generate the judgment_types.py file based on the schema in openapi_new.json.
datamodel-codegen --input judgeval/data/openapi_new.json --output judgeval/data/judgment_types.py --use-annotated

# Post-process the generated file to fix mutable defaults
uv run judgeval/data/scripts/fix_default_factory.py judgeval/data/judgment_types.py

# Remove the openapi_new.json file since it is no longer needed
rm judgeval/data/openapi_new.json