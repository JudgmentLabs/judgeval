import sys

if len(sys.argv) != 2:
    print("Usage: python set_version.py <new_version>")
    sys.exit(1)

new_version = sys.argv[1]
version_placeholder = "<version_placeholder>"
found = False

try:
    with open("pyproject.toml", "r") as f:
        lines = f.readlines()
except IOError as e:
    print(f"Error: Failed to read 'pyproject.toml': {e}")
    sys.exit(1)

try:
    with open("pyproject.toml", "w") as f:
        for line in lines: # Assumes 'lines' was successfully read earlier
            if not found and version_placeholder in line:
                f.write(line.replace(version_placeholder, new_version))
                found = True
            else:
                f.write(line)
except IOError as e:
    print(f"Error: Failed to write to 'pyproject.toml': {e}")
    sys.exit(1)

if not found:
    print("Warning: No '<version_placeholder>' found in pyproject.toml")
    sys.exit(1)
