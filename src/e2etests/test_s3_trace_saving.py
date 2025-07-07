import pytest
import boto3
import uuid
import asyncio
from botocore.exceptions import ClientError
from judgeval.tracer import Tracer
from unittest.mock import patch
import time

# Test constants
TEST_BUCKET_PREFIX = "judgeval-test-"
TEST_REGION = "us-west-1"  # Change this to your desired region


@pytest.fixture
def s3_client():
    """Create an S3 client for testing."""
    return boto3.client("s3", region_name=TEST_REGION)


@pytest.fixture
def test_bucket_name():
    """Generate a unique bucket name for testing."""
    return f"{TEST_BUCKET_PREFIX}{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_bucket(s3_client, test_bucket_name):
    """Create a temporary S3 bucket for testing."""
    try:
        s3_client.create_bucket(
            Bucket=test_bucket_name,
            CreateBucketConfiguration={"LocationConstraint": TEST_REGION},
        )
        yield test_bucket_name
    finally:
        # Clean up: delete all objects and then the bucket
        try:
            objects = s3_client.list_objects_v2(Bucket=test_bucket_name)
            if "Contents" in objects:
                delete_keys = {
                    "Objects": [{"Key": obj["Key"]} for obj in objects["Contents"]]
                }
                s3_client.delete_objects(Bucket=test_bucket_name, Delete=delete_keys)
            s3_client.delete_bucket(Bucket=test_bucket_name)
        except ClientError as e:
            print(f"Error cleaning up bucket {test_bucket_name}: {e}")


@pytest.fixture
def judgment(test_bucket, project_name: str):
    """Create a Tracer instance for testing."""
    Tracer._instance = None
    yield Tracer(
        project_name=project_name,
        s3_bucket_name=test_bucket,
        s3_region_name=TEST_REGION,
        use_s3=True,
    )
    Tracer._instance = None


@pytest.fixture
def judgment_no_bucket_yet(test_bucket_name, s3_client, project_name: str):
    Tracer._instance = None
    yield Tracer(
        project_name=project_name,
        s3_bucket_name=test_bucket_name,
        s3_region_name=TEST_REGION,
        use_s3=True,
    )
    Tracer._instance = None
    try:
        objects = s3_client.list_objects_v2(Bucket=test_bucket_name)
        if "Contents" in objects:
            delete_keys = {
                "Objects": [{"Key": obj["Key"]} for obj in objects["Contents"]]
            }
            s3_client.delete_objects(Bucket=test_bucket_name, Delete=delete_keys)
        s3_client.delete_bucket(Bucket=test_bucket_name)
    except ClientError as e:
        print(f"Error cleaning up bucket {test_bucket_name}: {e}")


@pytest.mark.asyncio
async def test_save_trace_to_s3(judgment, s3_client, project_name: str):
    """Test saving a trace to S3 using judgment.observe decorator."""

    test_output = "test output"

    @judgment.observe(name="test_trace")
    def test_function(input):
        return test_output

    # Call the decorated function
    test_function(input="test input")
    # Verify trace was saved to S3
    try:
        # List objects in the bucket
        response = s3_client.list_objects_v2(Bucket=judgment.s3_storage.bucket_name)
        assert "Contents" in response, "No objects found in bucket"

        # Find our trace file
        trace_files = [
            obj for obj in response["Contents"] if project_name in obj["Key"]
        ]
        assert len(trace_files) > 0, (
            "Trace file with ID project_name not found in bucket"
        )

        # Get the trace file content
        trace_file = trace_files[0]
        response = s3_client.get_object(
            Bucket=judgment.s3_storage.bucket_name, Key=trace_file["Key"]
        )
        trace_content = response["Body"].read().decode("utf-8")

        # Verify trace content
        assert test_output in trace_content
        assert "test input" in trace_content

    except ClientError as e:
        pytest.fail(f"Failed to verify trace in S3: {e}")


@pytest.mark.asyncio
async def test_auto_bucket_creation(
    judgment_no_bucket_yet, s3_client, project_name: str
):
    """Test that observe() automatically creates the S3 bucket if it doesn't exist."""

    # Verify bucket doesn't exist initially
    with pytest.raises(ClientError) as exc_info:
        s3_client.head_bucket(Bucket=judgment_no_bucket_yet.s3_storage.bucket_name)
    assert exc_info.value.response["Error"]["Code"] == "404"

    test_output = "test output"

    @judgment_no_bucket_yet.observe(name=project_name)
    def test_function(input):
        return test_output

    # Call the decorated function - this should create the bucket
    test_function(input="test input")

    # Poll for bucket creation with timeout
    timeout = 30  # 30 second timeout
    start_time = time.time()
    while True:
        try:
            s3_client.head_bucket(Bucket=judgment_no_bucket_yet.s3_storage.bucket_name)
            break  # Bucket exists, continue with test
        except ClientError as e:
            if time.time() - start_time > timeout:
                pytest.fail(
                    f"Bucket {judgment_no_bucket_yet.s3_storage.bucket_name} was not created after {timeout} seconds: {e}"
                )
            await asyncio.sleep(1)  # Wait 1 second before retrying

    # Verify trace was saved to S3
    try:
        # List objects in the bucket
        response = s3_client.list_objects_v2(
            Bucket=judgment_no_bucket_yet.s3_storage.bucket_name
        )
        assert "Contents" in response, "No objects found in bucket"

        # Find our trace file
        trace_files = [
            obj for obj in response["Contents"] if project_name in obj["Key"]
        ]
        assert len(trace_files) > 0, (
            "Trace file with ID project_name not found in bucket"
        )

        # Get the trace file content
        trace_file = trace_files[0]
        response = s3_client.get_object(
            Bucket=judgment_no_bucket_yet.s3_storage.bucket_name, Key=trace_file["Key"]
        )
        trace_content = response["Body"].read().decode("utf-8")

        # Verify trace content
        assert test_output in trace_content
        assert "test input" in trace_content

    except ClientError as e:
        pytest.fail(f"Failed to verify trace in S3: {e}")


@pytest.mark.asyncio
async def test_bucket_already_owned_by_you(judgment, s3_client):
    """Test handling of BucketAlreadyOwnedByYou error during bucket creation."""
    # Mock the S3 client to simulate BucketAlreadyOwnedByYou error
    with (
        patch.object(
            judgment.s3_storage.s3_client,
            "head_bucket",
            side_effect=ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
            ),
        ),
        patch.object(
            judgment.s3_storage.s3_client,
            "create_bucket",
            side_effect=ClientError(
                {
                    "Error": {
                        "Code": "BucketAlreadyOwnedByYou",
                        "Message": "Bucket already owned by you",
                    }
                },
                "CreateBucket",
            ),
        ),
    ):
        test_output = "test output"

        @judgment.observe(name="test_trace")
        def test_function(input):
            return test_output

        # Should not raise an error, should continue with existing bucket
        output = test_function(input="test input")
        assert output == test_output
