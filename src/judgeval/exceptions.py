from __future__ import annotations

from httpx import HTTPError, Response


class JudgmentAPIError(HTTPError):
    status_code: int
    detail: str
    response: Response

    def __init__(self, status_code: int, detail: str, response: Response):
        self.status_code = status_code
        self.detail = detail
        self.response = response
        super().__init__(f"{status_code}: {detail}")


class JudgmentTestError(Exception): ...


class JudgmentRuntimeError(RuntimeError): ...


class InvalidJudgeModelError(Exception): ...


__all__ = ("JudgmentAPIError", "JudgmentRuntimeError", "InvalidJudgeModelError")
