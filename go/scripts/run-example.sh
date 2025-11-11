#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: Please specify an example to run"
  echo "Usage: bun run example <example_name>"
  echo "Available examples:"
  ls -1 examples/ | grep -v "^\\." || echo "  (none found)"
  exit 1
fi

cd "examples/$1" && go run .

