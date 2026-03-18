from unittest.mock import patch, MagicMock

from judgeval.utils import version_check as vc


def test_spawns_daemon_thread():
    with patch("judgeval.utils.version_check.threading.Thread") as mock_thread_cls:
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        vc.check_latest_version.__wrapped__("judgeval")

        mock_thread_cls.assert_called_once()
        _, kwargs = mock_thread_cls.call_args
        assert kwargs.get("daemon") is True
        mock_thread.start.assert_called_once()


def test_check_silences_exception_when_network_fails():
    captured = {}

    def fake_thread(target, daemon):
        captured["target"] = target
        return MagicMock()

    with patch(
        "judgeval.utils.version_check.threading.Thread", side_effect=fake_thread
    ):
        with patch(
            "judgeval.utils.version_check.httpx.get",
            side_effect=Exception("network error"),
        ):
            with patch(
                "judgeval.utils.version_check.importlib.metadata.version",
                return_value="0.1.0",
            ):
                vc.check_latest_version.__wrapped__("judgeval")

    captured["target"]()  # must not raise


def test_use_once_prevents_second_invocation():
    call_log = []

    from judgeval.utils.decorators.use_once import use_once

    @use_once
    def fn():
        call_log.append(1)

    fn()
    fn()
    fn()
    assert len(call_log) == 1
