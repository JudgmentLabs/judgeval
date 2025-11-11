#!/usr/bin/env python3
"""
Post-process generated Pydantic models with default_factory defaults.
"""

import sys


def fix_mutable_defaults(file_path: str) -> None:
    """Fix mutable defaults in generated Pydantic models."""

    with open(file_path, "r") as f:
        content = f.read()

    content = content.replace(" = {}", " = Field(default_factory=dict)")
    content = content.replace(" = []", " = Field(default_factory=list)")
    with open(file_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    file_path = sys.argv[1]
    fix_mutable_defaults(file_path)
