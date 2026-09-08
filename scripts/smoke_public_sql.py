"""Run the Doe production SQL smoke query using an explicitly supplied env file.

Usage: uv run python scripts/smoke_public_sql.py /path/to/.env.prod
Never prints credentials or raw error responses.
"""

import json
import sys

import httpx
from dotenv import dotenv_values

SQL = """SELECT count() AS run_count
FROM telemetry.traces
WHERE started_at >= toDateTime64('2026-09-01 00:00:00', 6)
  AND started_at < toDateTime64('2026-09-04 00:00:00', 6)"""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: smoke_public_sql.py /path/to/.env.prod")
    config = dotenv_values(sys.argv[1])
    api_key = config.get("JUDGMENT_API_KEY")
    api_url = config.get("JUDGMENT_API_URL")
    if not api_key or not api_url:
        raise SystemExit("JUDGMENT_API_KEY and JUDGMENT_API_URL are required")
    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}/v1/projects/4ed11e98-a948-4ae7-9696-3a8e61636845/sql",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Organization-Id": "901ab31a-5280-4d26-bf0f-93b4e0e0f78c",
            },
            json={"sql": SQL},
            timeout=120,
        )
    except httpx.HTTPError:
        raise SystemExit("Public SQL request failed at the transport layer") from None
    if response.status_code != 200:
        raise SystemExit(f"Public SQL returned HTTP {response.status_code}")
    result = response.json()
    rows = result["rows"]
    count = int(rows[0]["run_count"])
    if rows != [{"run_count": rows[0]["run_count"]}] or result["row_count"] != 1:
        raise SystemExit("Unexpected count-query response shape")
    print(
        json.dumps(
            {
                "run_count": count,
                "previous_count": 64376,
                "matches_previous": count == 64376,
            }
        )
    )


if __name__ == "__main__":
    main()
