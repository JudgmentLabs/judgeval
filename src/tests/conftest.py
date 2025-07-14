import pytest
import requests
import httpx
import urllib.request
import inspect


@pytest.fixture(autouse=True)
def block_http_requests(monkeypatch):
    """Block any HTTP requests that aren't explicitly mocked."""

    def get_error_info():
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        return inspect.getframeinfo(caller_frame)

    def is_external_url(url):
        return True

    # Store original methods
    original_requests = {
        "get": requests.get,
        "post": requests.post,
        "put": requests.put,
        "delete": requests.delete,
        "head": requests.head,
        "patch": requests.patch,
    }

    original_httpx = {
        "get": httpx.get,
        "post": httpx.post,
        "put": httpx.put,
        "delete": httpx.delete,
        "head": httpx.head,
        "patch": httpx.patch,
        "request": httpx.request,
    }

    original_httpx_client = {
        method: getattr(httpx.Client, method) for method in original_httpx
    }

    original_httpx_async_client = {
        method: getattr(httpx.AsyncClient, method) for method in original_httpx
    }

    original_urllib = urllib.request.urlopen

    def make_blocked_requests(method_name, original_func):
        def wrapper(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            if not is_external_url(url):
                return original_func(*args, **kwargs)
            caller = get_error_info()
            raise RuntimeError(
                f"Blocked requests.{method_name.upper()} to {url}\n"
                f"Called from: {caller.filename}:{caller.lineno} in {caller.function}\n"
                f"Please mock this request in your test."
            )

        return wrapper

    def make_blocked_httpx_module(method_name, original_func):
        def wrapper(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            if not is_external_url(url):
                filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                return original_func(*args, **filtered_kwargs)
            caller = get_error_info()
            raise RuntimeError(
                f"Blocked httpx.{method_name.upper()} to {url}\n"
                f"Called from: {caller.filename}:{caller.lineno} in {caller.function}\n"
                f"Please mock this request in your test."
            )

        return wrapper

    def make_blocked_httpx_client(method_name, original_func):
        def wrapper(self, *args, **kwargs):
            from httpx import URL

            if method_name == "request" and len(args) >= 2:
                url_arg = args[1]
            else:
                url_arg = args[0] if args else kwargs.get("url", "")

            if isinstance(url_arg, str) and hasattr(self, "base_url") and self.base_url:
                full_url = URL(self.base_url).join(URL(url_arg))
            else:
                full_url = url_arg

            if not is_external_url(str(full_url)):
                filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                return original_func(self, *args, **filtered_kwargs)

            caller = get_error_info()
            raise RuntimeError(
                f"Blocked httpx.Client.{method_name.upper()} to {full_url}\n"
                f"Called from: {caller.filename}:{caller.lineno} in {caller.function}\n"
                f"Please mock this request in your test."
            )

        return wrapper

    def make_blocked_httpx_async_client(method_name, original_func):
        async def wrapper(self, *args, **kwargs):
            from httpx import URL

            # Extract method and url argument based on call signature
            if method_name == "request" and len(args) >= 2:
                url_arg = args[1]
            else:
                url_arg = args[0] if args else kwargs.get("url", "")

            # Reconstruct full URL
            if isinstance(url_arg, str) and hasattr(self, "base_url") and self.base_url:
                full_url = URL(self.base_url).join(URL(url_arg))
            else:
                full_url = url_arg

            if not is_external_url(str(full_url)):
                filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                return await original_func(self, *args, **filtered_kwargs)

            caller = get_error_info()
            raise RuntimeError(
                f"Blocked httpx.AsyncClient.{method_name.upper()} to {full_url}\n"
                f"Called from: {caller.filename}:{caller.lineno} in {caller.function}\n"
                f"Please mock this request in your test."
            )

        return wrapper

    def blocked_urllib(*args, **kwargs):
        url = args[0] if args else kwargs.get("url", "")
        if not is_external_url(url):
            return original_urllib(*args, **kwargs)
        caller = get_error_info()
        raise RuntimeError(
            f"Blocked urllib request to {url}\n"
            f"Called from: {caller.filename}:{caller.lineno} in {caller.function}\n"
            f"Please mock this request in your test."
        )

    # Patch requests
    for method, original_func in original_requests.items():
        monkeypatch.setattr(
            requests, method, make_blocked_requests(method, original_func)
        )

    # Patch httpx module-level functions
    for method, original_func in original_httpx.items():
        monkeypatch.setattr(
            httpx, method, make_blocked_httpx_module(method, original_func)
        )

    # Patch httpx.Client instance methods
    for method, original_func in original_httpx_client.items():
        monkeypatch.setattr(
            httpx.Client, method, make_blocked_httpx_client(method, original_func)
        )

    # Patch httpx.AsyncClient instance methods
    for method, original_func in original_httpx_async_client.items():
        monkeypatch.setattr(
            httpx.AsyncClient,
            method,
            make_blocked_httpx_async_client(method, original_func),
        )

    # Patch urllib
    monkeypatch.setattr(urllib.request, "urlopen", blocked_urllib)
