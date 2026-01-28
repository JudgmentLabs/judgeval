"""
Reliability test suite for the v1 SDK.

These tests verify the SDK is safe for high-throughput customer deployments by testing:
- Latency overhead
- Memory stability
- Failure isolation
- Thread safety
- Edge case handling

Run with: pytest -m reliability src/tests/reliability/ -v
"""
