from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Literal, Sequence, Optional, TYPE_CHECKING, cast
import boto3
from botocore.client import Config

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from judgeval.env import (
    JUDGMENT_S3_ACCESS_KEY_ID,
    JUDGMENT_S3_SECRET_ACCESS_KEY,
    JUDGMENT_S3_REGION_NAME,
    JUDGMENT_S3_BUCKET_NAME,
    JUDGMENT_S3_PREFIX,
    JUDGMENT_S3_ENDPOINT_URL,
    JUDGMENT_S3_SIGNATURE_VERSION,
    JUDGMENT_S3_ADDRESSING_STYLE,
)
from judgeval.exceptions import JudgmentRuntimeError
from judgeval.logger import judgeval_logger

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


class S3Exporter(SpanExporter):
    __slots__ = ("bucket_name", "prefix", "s3_client")

    bucket_name: str
    prefix: str
    s3_client: S3Client

    def __init__(
        self,
        bucket_name: Optional[str] = JUDGMENT_S3_BUCKET_NAME,
        region_name: Optional[str] = JUDGMENT_S3_REGION_NAME,
        prefix: str = JUDGMENT_S3_PREFIX,
        s3_access_key_id: Optional[str] = JUDGMENT_S3_ACCESS_KEY_ID,
        s3_secret_access_key: Optional[str] = JUDGMENT_S3_SECRET_ACCESS_KEY,
        endpoint_url: Optional[str] = JUDGMENT_S3_ENDPOINT_URL,
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
        signature_version: str = JUDGMENT_S3_SIGNATURE_VERSION,
        addressing_style: str = JUDGMENT_S3_ADDRESSING_STYLE,
        batch_size: int = 8,
    ):
        if not bucket_name:
            raise JudgmentRuntimeError("JUDGMENT_S3_BUCKET_NAME is not set")

        if not region_name:
            raise JudgmentRuntimeError("JUDGMENT_S3_REGION_NAME is not set")

        if addressing_style not in ["auto", "virtual", "path"]:
            raise JudgmentRuntimeError(f"Invalid addressing style: {addressing_style}")
        addressing_style = cast(Literal["auto", "virtual", "path"], addressing_style)

        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/")
        self.batch_size = batch_size

        self.s3_client = boto3.client(
            "s3",
            config=Config(
                signature_version=signature_version,
                s3={"addressing_style": addressing_style},
            ),
            aws_access_key_id=s3_access_key_id,
            aws_secret_access_key=s3_secret_access_key,
            endpoint_url=endpoint_url,
            region_name=region_name,
        )

    def _upload_span(self, span: ReadableSpan) -> tuple[bool, str]:
        """Upload a single span to S3. Returns (success, key)."""
        try:
            span_context = span.get_span_context()
            if not span_context:
                return False, ""

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            trace_id = format(span_context.trace_id, "032x")
            span_id = format(span_context.span_id, "016x")
            key = f"{self.prefix}/{trace_id}/{span_id}/{timestamp}.json"

            span_json = span.to_json(indent=0)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=span_json,
                ContentType="application/json",
            )
            return True, key
        except Exception as e:
            judgeval_logger.error(
                f"Error uploading span {span_context.span_id if span_context else 'unknown'}: {e}"
            )
            return False, ""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS

        try:
            with ThreadPoolExecutor(
                max_workers=min(len(spans), self.batch_size)
            ) as executor:
                futures = [executor.submit(self._upload_span, span) for span in spans]

                for future in as_completed(futures):
                    success, key = future.result()
                    if not success:
                        return SpanExportResult.FAILURE
                return SpanExportResult.SUCCESS

        except Exception as e:
            judgeval_logger.error(f"Error exporting spans to S3: {e}")
            return SpanExportResult.FAILURE
