from requests import exceptions
from judgeval.utils.requests import requests, requests_original
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import pytest
import threading
import time


@pytest.fixture
def http_server_fixture():
    class CustomHandler(BaseHTTPRequestHandler):
        call_count = 0

        def do_GET(self):
            if self.path == "/get_success":
                CustomHandler.call_count += 1
                print(f"Call count: {CustomHandler.call_count}")
                if CustomHandler.call_count == 2:
                    self.send_response(502)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    response = {
                        "status": "error",
                        "call": CustomHandler.call_count,
                        "code": 502,
                    }
                    self.wfile.write(json.dumps(response).encode())
                if CustomHandler.call_count == 3:
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    response = {
                        "status": "success",
                        "call": CustomHandler.call_count,
                        "code": 200,
                    }
                    self.wfile.write(json.dumps(response).encode())
                else:
                    self.send_response(503)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    response = {
                        "status": "error",
                        "call": CustomHandler.call_count,
                        "code": 503,
                    }
                    self.wfile.write(json.dumps(response).encode())
            elif self.path == "/get_error":
                CustomHandler.call_count += 1
                print(f"Call count: {CustomHandler.call_count}")
                self.send_response(503)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {
                    "status": "error",
                    "call": CustomHandler.call_count,
                    "code": 503,
                }
                self.wfile.write(json.dumps(response).encode())
            elif self.path == "/get_500":
                CustomHandler.call_count += 1
                print(f"Call count: {CustomHandler.call_count}")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {
                    "status": "error",
                    "call": CustomHandler.call_count,
                    "code": 500,
                }
                self.wfile.write(json.dumps(response).encode())
            elif self.path == "/health":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")

    server = HTTPServer(("localhost", 8002), CustomHandler)

    CustomHandler.call_count = 0

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    while requests_original.get("http://localhost:8002/health").status_code != 200:
        time.sleep(0.1)

    yield server

    # Cleanup
    server.shutdown()
    server.server_close()
    server_thread.join(timeout=1)


def test_requests_backoff(http_server_fixture):
    response = requests.get("http://localhost:8002/get_success")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    assert data["status"] == "success"
    assert data["call"] == 3
    assert data["code"] == 200


def test_requests_backoff_limit(http_server_fixture):
    with pytest.raises(exceptions.RetryError):
        requests.get("http://localhost:8002/get_error")


def test_requests_backoff_non_502_503(http_server_fixture):
    response = requests.get("http://localhost:8002/get_501")
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"

    # No retry on error codes other than 502 and 503
    data = response.json()
    assert data["status"] == "error"
    assert data["call"] == 1
    assert data["code"] == 404
