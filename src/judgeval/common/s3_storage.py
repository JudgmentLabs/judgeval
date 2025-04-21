import os
import json
import boto3
from typing import Optional
from datetime import datetime, UTC

class S3Storage:
    """Utility class for storing and retrieving trace data from S3."""
    
    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None
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
            's3',
            aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region_name or os.getenv('AWS_REGION', 'us-west-1')
        )
        
    def save_trace(self, trace_data: dict, trace_id: str, project_name: str) -> str:
        """Save trace data to S3.
        
        Args:
            trace_data: The trace data to save
            trace_id: Unique identifier for the trace
            project_name: Name of the project the trace belongs to
            
        Returns:
            str: S3 key where the trace was saved
        """
        # Create a timestamped key for the trace
        timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
        s3_key = f"traces/{project_name}/{trace_id}_{timestamp}.json"
        
        # Convert trace data to JSON string
        trace_json = json.dumps(trace_data)
        
        # Upload to S3
        print(f"Uploading trace to S3 at key {s3_key}, in bucket {self.bucket_name} ...")
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=trace_json,
            ContentType='application/json'
        )
        
        return s3_key
        
    def get_trace(self, s3_key: str) -> dict:
        """Retrieve trace data from S3.
        
        Args:
            s3_key: S3 key where the trace is stored
            
        Returns:
            dict: The trace data
        """
        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=s3_key
        )
        
        trace_json = response['Body'].read().decode('utf-8')
        return json.loads(trace_json) 