import sys

if len(sys.argv) != 2:
    print("Usage: python set_version.py <new_version>")
    sys.exit(1)

new_version = sys.argv[1]
version_placeholder = "0.0.0"


def update_file(file_path, placeholder, new_val):
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except IOError as e:
        print(f"Error: Failed to read '{file_path}': {e}")
        sys.exit(1)

    if placeholder not in content:
        print(f"Warning: No version placeholder '{placeholder}' found in '{file_path}'")
        sys.exit(1)

    new_content = content.replace(placeholder, new_val, 1)

    try:
        with open(file_path, "w") as f:
            f.write(new_content)
    except IOError as e:
        print(f"Error: Failed to write to '{file_path}': {e}")
        sys.exit(1)


files_to_update = ["pyproject.toml", "src/judgeval/version.py"]
for file in files_to_update:
    update_file(file, version_placeholder, new_version)
