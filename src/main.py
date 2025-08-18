import os

from opentelemetry.sdk.trace.export import BatchSpanProcessor
from judgeval.tracer import Tracer
from judgeval.tracer.exporters import S3Exporter

exporter = S3Exporter(
    bucket_name="ahh-judgment-staging-test-bucket",
    prefix="test/",
    endpoint_url="https://storage.googleapis.com",
)

try:
    response = exporter.s3_client.head_bucket(Bucket="ahh-judgment-staging-test-bucket")
    print(response)
    print("✓ Bucket access successful")
except Exception as e:
    print(f"✗ Bucket access failed: {e}")

tracer = Tracer(
    project_name="ahh",
    processors=[BatchSpanProcessor(exporter)],
)


@tracer.observe
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


fib(10)
