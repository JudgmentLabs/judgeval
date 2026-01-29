#!/usr/bin/env python3
"""
Quick script to test trace fetching endpoint.

Usage:
    export JUDGMENT_API_KEY="your-key"
    export JUDGMENT_ORG_ID="your-org-id"
    export JUDGMENT_API_URL="https://staging.judgmentlabs.ai"

    python src/tests/reliability/test_trace_fetch.py <project_id>
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp not installed. Run: pip install aiohttp")
    sys.exit(1)


def iso_now_minus(minutes: int) -> str:
    """Get ISO timestamp N minutes ago."""
    return (datetime.utcnow() - timedelta(minutes=minutes)).replace(
        microsecond=0
    ).isoformat() + "Z"


async def fetch_traces(
    base_url: str,
    access_token: str,
    org_id: str,
    project_id: str,
    limit: int = 100,
    start_time: Optional[str] = None,
):
    """Fetch traces from the API."""

    # Build URL
    path = f"/projects/{project_id}/traces"
    url = f"{base_url.rstrip('/')}{path}/"

    # if start_time:
    #     url += f"?limit={limit}&start_time={start_time}"
    # else:
    #     url += f"?limit={limit}"

    print(f"\n{'=' * 80}")
    print("Fetching traces from endpoint:")
    print(f"  URL: {url}")
    print(f"  Project ID: {project_id}")
    print(f"  Org ID: {org_id}")
    print(f"{'=' * 80}\n")

    headers = {
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate, br",
        "Authorization": f"Bearer {access_token}",
        "X-Organization-Id": org_id,
    }

    body = {"filters": []}

    if start_time:
        body["time_range"] = {
            "start_time": start_time,
            "end_time": None,
        }
        body["pagination"] = {
            "limit": limit,
            "cursorCreatedAt": None,
            "cursorItemId": None,
        }

    print(f"Headers: {headers}")
    print(f"Body: {body}")
    print(f"URL: {url}")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body) as resp:
            print(f"Response Status: {resp.status} {resp.reason}")
            print(f"Response Headers: {dict(resp.headers)}\n")

            if resp.status == 401:
                print("❌ Unauthorized: Invalid or expired access token")
                return None

            if resp.status == 403:
                print("❌ Forbidden: Insufficient permissions")
                return None

            if resp.status == 404:
                print(f"❌ Not Found: Project {project_id} does not exist")
                return None

            if resp.status < 200 or resp.status >= 300:
                text = await resp.text()
                print(f"❌ API request failed: {resp.status} {resp.reason}")
                print(f"Response body: {text}")
                return None

            data = await resp.json()
            traces = data.get("data", [])

            print(f"✅ Success! Retrieved {len(traces)} traces")

            if traces:
                print("\nFirst trace sample:")
                print(f"  Trace ID: {traces[0].get('trace_id', 'N/A')}")
                print(f"  Span Name: {traces[0].get('span_name', 'N/A')}")
                print(f"  Span ID: {traces[0].get('span_id', 'N/A')}")
                print(f"  Timestamp: {traces[0].get('timestamp', 'N/A')}")

                # Show all keys in the first trace
                print(f"\n  Available keys: {list(traces[0].keys())}")
            else:
                print("\nNo traces found in the response.")

            return traces


async def main():
    # Get environment variables
    api_key = os.environ.get("JUDGMENT_API_KEY", "280353d5-f015-4dfe-86d2-e67ff16a1747")
    org_id = os.environ.get("JUDGMENT_ORG_ID", "df8607d9-1467-4e5e-b33d-c23e52a161fc")
    # api_url = os.environ.get("JUDGMENT_API_URL", "https://staging.judgmentlabs.ai")
    api_url = "https://api.judgmentlabs.ai"

    if not api_key:
        print("Error: JUDGMENT_API_KEY environment variable not set")
        sys.exit(1)

    if not org_id:
        print("Error: JUDGMENT_ORG_ID environment variable not set")
        sys.exit(1)

    # Get project_id from command line
    if len(sys.argv) < 2:
        print("Usage: python test_trace_fetch.py <project_id>")
        print("\nExample:")
        print("  python test_trace_fetch.py proj_abc123xyz")
        sys.exit(1)

    project_id = sys.argv[1]

    # Fetch traces from last 60 minutes
    start_time = iso_now_minus(60)

    traces = await fetch_traces(
        base_url=api_url,
        access_token=api_key,
        org_id=org_id,
        project_id=project_id,
        limit=100,
        start_time=start_time,
    )

    if traces is not None:
        print(f"\n{'=' * 80}")
        print(f"Summary: Found {len(traces)} traces in the last 60 minutes")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
