from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

import judgeval.conversations.conversations as conversations_module
from judgeval.conversations import Conversations, Observation


class FakeProcessor:
    def __init__(self) -> None:
        self.emitted: list[Any] = []

    def emit(self, observation: Any) -> None:
        self.emitted.append(observation)

    def force_flush(self) -> bool:
        return True

    def shutdown(self, timeout_seconds: float = 30.0) -> None:
        pass


def call(conversation_id: str = "ticket-1") -> Observation:
    return {
        "conversation_id": conversation_id,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}],
            }
        ],
    }


@pytest.fixture(autouse=True)
def _reset():
    yield
    Conversations._initialized = False
    Conversations._processor = None
    Conversations._environment = None


def use(environment: Optional[str] = None) -> FakeProcessor:
    """Makes a fake processor the destination, as `Conversations.init` would."""
    processor = FakeProcessor()
    Conversations._initialized = True
    Conversations._processor = processor  # type: ignore[assignment]
    Conversations._environment = environment
    return processor


def test_records_a_call_with_its_time_and_the_configured_environment():
    processor = use(environment="production")
    observed_at = datetime(2026, 11, 11, 12, 0, 1, tzinfo=timezone.utc)
    Conversations.record(call(), observed_at=observed_at)
    assert processor.emitted == [
        {
            "observation": call(),
            "observed_at": "2026-11-11T12:00:01+00:00",
            "environment": "production",
        }
    ]


def test_reads_a_naive_time_as_local_time():
    processor = use()
    local = datetime(2026, 11, 11, 12, 0, 1)
    Conversations.record(call(), observed_at=local)
    sent_at = datetime.fromisoformat(processor.emitted[0]["observed_at"])
    assert (sent_at.utcoffset(), sent_at) == (timedelta(0), local.astimezone())


def test_sends_the_call_as_it_was_when_recorded():
    processor = use()
    recorded = call()
    Conversations.record(recorded)
    recorded["conversation_id"] = "changed"
    assert processor.emitted[0]["observation"]["conversation_id"] == "ticket-1"


def test_recording_before_init_does_nothing():
    Conversations.record(call())
    assert Conversations.flush()


def test_init_without_credentials_leaves_recording_off(monkeypatch):
    monkeypatch.setattr(conversations_module, "JUDGMENT_API_KEY", None)
    Conversations.init(project_name="support-bot")
    Conversations.record(call())
    assert (Conversations._initialized, Conversations._processor) == (True, None)
