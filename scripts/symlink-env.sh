#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || exit 1

if [ ! -f .env ]; then
    echo "Error: .env file not found in $ROOT_DIR"
    exit 1
fi

languages=("python" "js" "java" "go")
for language in ${languages[@]}; do
    ln -sf "../.env" "$language/.env"
    echo "Linked .env to $language/.env"
done
