import os
import json
import boto3
from typing import Optional
from datetime import datetime, UTC
from botocore.exceptions import ClientError
from judgeval.common.logger import judgeval_logger


class S3Storage:
    """Utility class for storing and retrieving trace data from S3."""

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """Initialize S3 storage with credentials and bucket name.

        Args:
            bucket_name: Name of the S3 bucket to store traces in
            aws_access_key_id: AWS access key ID (optional, will use environment variables if not provided)
            aws_secret_access_key: AWS secret access key (optional, will use environment variables if not provided)
            region_name: AWS region name (optional, will use environment variables if not provided)
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name or os.getenv("AWS_REGION", "us-west-1"),
        )

    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, creating it if necessary."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                # Bucket doesn't exist, create it
                try:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={
                            "LocationConstraint": self.s3_client.meta.region_name
                        },
                    ) if self.s3_client.meta.region_name != "us-east-1" else self.s3_client.create_bucket(
                        Bucket=self.bucket_name
                    )
                except ClientError as create_error:
                    if (
                        create_error.response["Error"]["Code"]
                        == "BucketAlreadyOwnedByYou"
                    ):
                        # Bucket was just created by another process
                        judgeval_logger.warning(
                            f"Bucket {self.bucket_name} was just created by another process"
                        )
                        pass
                    else:
                        raise create_error
            else:
                # Some other error occurred
                raise e

    def save_trace(self, trace_data: dict, trace_id: str, project_name: str) -> str:
        """Save trace data to S3.

        Args:
            trace_data: The trace data to save
            trace_id: Unique identifier for the trace
            project_name: Name of the project the trace belongs to

        Returns:
            str: S3 key where the trace was saved
        """
        # Ensure bucket exists before saving
        self._ensure_bucket_exists()

        # Create a timestamped key for the trace
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        s3_key = f"traces/{project_name}/{trace_id}_{timestamp}.json"

        # Convert trace data to JSON string
        trace_json = json.dumps(trace_data)

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=trace_json,
            ContentType="application/json",
        )

        return s3_key
