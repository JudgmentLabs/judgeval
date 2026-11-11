from __future__ import annotations

import copy
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pytest

import judgeval.conversations.conversations as conversations_module
from judgeval.conversations import Conversations, Observation
from judgeval.exceptions import JudgmentAPIError


class FakeClient:
    """Records each batch sent; `respond` decides each request's outcome."""

    def __init__(self, respond: Optional[Callable[[list[Any]], None]] = None):
        self.batches: list[list[Any]] = []
        self.attempts = 0
        self._respond = respond

    def post_projects_observations(self, project_id: str, payload: Any) -> Any:
        self.attempts += 1
        batch = copy.deepcopy(payload["observations"])
        if self._respond is not None:
            self._respond(batch)
        self.batches.append(batch)
        return {"status": "success"}


def api_error(status: int) -> JudgmentAPIError:
    return JudgmentAPIError(status, "error", None)


def call(conversation_id: str = "ticket-1", text: str = "Hi") -> Observation:
    return {
        "conversation_id": conversation_id,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(conversations_module, "FLUSH_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(conversations_module.time, "sleep", lambda _: None)
    yield
    Conversations.shutdown(timeout_seconds=1)
    Conversations._active = None


def use(fake: FakeClient, environment: Optional[str] = None) -> None:
    """Makes `fake` the active destination, as `Conversations.init` would."""
    Conversations._active = Conversations(fake, "project-1", environment)  # type: ignore[arg-type]


def test_sends_a_recorded_call_with_its_time_and_environment():
    fake = FakeClient()
    use(fake, environment="production")
    observed_at = datetime(2026, 9, 25, 12, 0, 1, tzinfo=timezone.utc)
    Conversations.record(call(), observed_at=observed_at)
    assert Conversations.flush(timeout_seconds=5)
    assert fake.batches == [
        [
            {
                "observation": call(),
                "observed_at": "2026-09-25T12:00:01+00:00",
                "environment": "production",
            }
        ]
    ]


def test_sends_at_most_one_hundred_calls_per_request():
    fake = FakeClient()
    use(fake)
    for index in range(250):
        Conversations.record(call(text=str(index)))
    assert Conversations.flush(timeout_seconds=5)
    assert [len(batch) for batch in fake.batches] == [100, 100, 50]


def test_a_send_failure_never_reaches_the_caller():
    def fail(batch: list[Any]) -> None:
        raise RuntimeError("network down")

    fake = FakeClient(fail)
    use(fake)
    Conversations.record(call())
    assert Conversations.flush(timeout_seconds=5)
    assert (fake.batches, fake.attempts) == ([], conversations_module.MAX_ATTEMPTS)


def test_retries_a_temporary_failure():
    failures = [api_error(503)]

    def fail_once(batch: list[Any]) -> None:
        if failures:
            raise failures.pop()

    fake = FakeClient(fail_once)
    use(fake)
    Conversations.record(call())
    assert Conversations.flush(timeout_seconds=5)
    assert (len(fake.batches), fake.attempts) == (1, 2)


def test_a_call_that_does_not_fit_the_format_does_not_drop_the_others():
    def reject_bad(batch: list[Any]) -> None:
        if any(sent["observation"]["conversation_id"] == "bad" for sent in batch):
            raise api_error(400)

    fake = FakeClient(reject_bad)
    use(fake)
    for conversation_id in ["a", "bad", "b", "c"]:
        Conversations.record(call(conversation_id))
    assert Conversations.flush(timeout_seconds=5)
    sent = sorted(
        sent["observation"]["conversation_id"]
        for batch in fake.batches
        for sent in batch
    )
    assert sent == ["a", "b", "c"]


@pytest.mark.parametrize("status", [402, 404])
def test_drops_calls_without_retrying_when_the_server_will_not_store_them(status):
    def refuse(batch: list[Any]) -> None:
        raise api_error(status)

    fake = FakeClient(refuse)
    use(fake)
    Conversations.record(call())
    assert Conversations.flush(timeout_seconds=5)
    assert fake.attempts == 1


def test_drops_calls_once_the_queue_is_full(monkeypatch):
    monkeypatch.setattr(conversations_module, "MAX_QUEUE_SIZE", 3)
    release = threading.Event()
    fake = FakeClient(lambda batch: release.wait(5))
    use(fake)
    Conversations.record(call(text="sending"))
    # Wait until the first call is taken off the queue and being sent.
    for _ in range(500):
        if fake.attempts:
            break
        threading.Event().wait(0.01)
    for index in range(5):
        Conversations.record(call(text=str(index)))
    release.set()
    assert Conversations.flush(timeout_seconds=5)
    assert sum(len(batch) for batch in fake.batches) == 4


def test_reads_a_naive_time_as_local_time():
    fake = FakeClient()
    use(fake)
    local = datetime(2026, 9, 25, 12, 0, 1)
    Conversations.record(call(), observed_at=local)
    assert Conversations.flush(timeout_seconds=5)
    sent_at = datetime.fromisoformat(fake.batches[0][0]["observed_at"])
    assert (sent_at.utcoffset(), sent_at) == (timedelta(0), local.astimezone())


def test_sends_the_call_as_it_was_when_recorded():
    fake = FakeClient()
    use(fake)
    recorded = call()
    Conversations.record(recorded)
    recorded["conversation_id"] = "changed"
    assert Conversations.flush(timeout_seconds=5)
    assert fake.batches[0][0]["observation"]["conversation_id"] == "ticket-1"


def test_ignores_calls_recorded_after_shutdown():
    fake = FakeClient()
    use(fake)
    Conversations.shutdown(timeout_seconds=1)
    Conversations.record(call())
    assert Conversations.flush(timeout_seconds=1)
    assert fake.attempts == 0


def test_recording_before_init_does_nothing():
    Conversations.record(call())
    assert Conversations.flush(timeout_seconds=1)


def test_init_without_credentials_leaves_recording_off(monkeypatch):
    monkeypatch.setattr(conversations_module, "JUDGMENT_API_KEY", None)
    Conversations.init(project_name="support-bot")
    Conversations.record(call())
    assert Conversations.flush(timeout_seconds=1)
    assert Conversations._active is not None
    assert Conversations._active._client is None


def test_a_forked_child_starts_with_an_empty_sender():
    fake = FakeClient(lambda batch: None)
    use(fake)
    active = Conversations._active
    assert active is not None
    active._queue.append({"observation": call(), "observed_at": "now"})
    conversations_module._after_fork_in_child()
    assert (len(active._queue), active._thread) == (0, None)
