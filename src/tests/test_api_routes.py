import json
import re
from pathlib import Path

import judgeval.internal.api.api_client as api_client

SPEC_PATH = Path(__file__).resolve().parents[2] / "openapi" / "public-api.json"


def _route_key(method: str, path: str) -> tuple[str, str]:
    return method.upper(), re.sub(r"\{[^}]+\}", "{}", path)


def test_client_routes_exist_in_committed_spec():
    spec = json.loads(SPEC_PATH.read_text())
    spec_routes = {
        _route_key(method, path)
        for path, operations in spec["paths"].items()
        for method in operations
    }
    source = Path(api_client.__file__).read_text()
    client_routes = {
        _route_key(method, path)
        for method, path in re.findall(
            r'"(GET|POST|PATCH|PUT|DELETE)",\s*url_for\(\s*f?"([^"]+)"', source
        )
    }

    assert (len(client_routes), sorted(client_routes - spec_routes)) == (28, [])
