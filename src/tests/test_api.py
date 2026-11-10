import json

import httpx

from judgeval.api import JudgmentApiClient


def test_generated_method_sends_method_path_and_body():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "j1"})

    client = JudgmentApiClient("https://api.test", "key", "org")
    client.client = httpx.Client(transport=httpx.MockTransport(handler))

    assert client.update_judge(
        "p1", "j1", {"update": {"judgeDescription": "tone"}}
    ) == {"id": "j1"}
    [request] = requests
    assert (request.method, request.url.path, json.loads(request.content)) == (
        "PATCH",
        "/v1/projects/p1/judges/j1",
        {"update": {"judgeDescription": "tone"}},
    )


def test_generated_method_encodes_path_parameters():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    client = JudgmentApiClient("https://api.test", "key", "org")
    client.client = httpx.Client(transport=httpx.MockTransport(handler))

    client.delete_judge("//evil.test", "../test-configs/x")
    [request] = requests
    assert (request.url.host, request.url.raw_path) == (
        "api.test",
        b"/v1/projects/%2F%2Fevil.test/judges/..%2Ftest-configs%2Fx",
    )


def test_previous_judge_configuration_methods_keep_their_http_contracts():
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={})

    client = JudgmentApiClient("https://api.test", "key", "org")
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    client.set_judge_optimistic_eval_enabled("p", "j", "v", {"enabled": True})
    client.set_judge_preprocessor_artifact_revision(
        "p", "j", "v", {"preprocessor_artifact_revision_id": None}
    )
    assert requests == [
        (
            "PUT",
            "/v1/projects/p/judges/j/versions/v/optimistic-eval",
            {"enabled": True},
        ),
        (
            "PUT",
            "/v1/projects/p/judges/j/versions/v/preprocessor-artifact-revision",
            {"preprocessor_artifact_revision_id": None},
        ),
    ]
