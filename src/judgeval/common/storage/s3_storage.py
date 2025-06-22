import os
import json
import dataclasses
from typing import Optional
from datetime import datetime, UTC

import logging
from judgeval.common.storage.storage import ABCStorage
from judgeval.common.tracer.model import TraceSave

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]


class S3Storage(ABCStorage):
    __slots__ = ("bucket_name", "s3_client")

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        use_ssl: bool = True,
        path_style: bool = False,
    ):
        super().__init__()
        self.bucket_name = bucket_name

        config = {
            "aws_access_key_id": access_key_id
            or os.getenv("S3_ACCESS_KEY_ID")
            or os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": secret_access_key
            or os.getenv("S3_SECRET_ACCESS_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
            "use_ssl": use_ssl,
        }

        if endpoint_url:
            config["endpoint_url"] = endpoint_url
        if region_name:
            config["region_name"] = region_name
        if path_style:
            config["config"] = Config(s3={"addressing_style": "path"})

        self.s3_client = boto3.client("s3", **config)

    def _ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logging.info(f"Created bucket: {self.bucket_name}")
                except ClientError as ce:
                    if ce.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
                        logging.warning(f"Bucket {self.bucket_name} already exists")
                    else:
                        raise
            elif code == "403":
                raise PermissionError(f"Access denied to bucket {self.bucket_name}")
            else:
                raise

    def save_trace(
        self, trace_data: TraceSave, trace_id: str, project_name: str
    ) -> str:
        self._ensure_bucket_exists()
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        key = f"traces/{project_name}/{trace_id}_{timestamp}.json"
        body = trace_data.model_dump_json()

        logging.info(f"Uploading trace to {key} in bucket {self.bucket_name}")
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return key
