"""
Locust load testing for Judgeval SDK.

Run with:
    locust -f src/tests/reliability/load_tests/locustfile.py

Then open http://localhost:8089 and configure:
    - Number of users: 1000
    - Spawn rate: 100/sec
    - Duration: 10 minutes
"""

import time

from locust import User, task, between, events

from judgeval.v1 import Judgeval


class TracerUser(User):
    """Simulates a customer application using Judgeval SDK."""

    wait_time = between(0.01, 0.1)

    def on_start(self):
        judgeval = Judgeval()
        self.tracer = judgeval.tracer.create(
            project_name="load-test",
            enable_monitoring=True,
            enable_evaluation=False,
            isolated=True,
        )
        self.iteration = 0

    @task(10)
    def trace_simple_function(self):
        start = time.perf_counter()

        @self.tracer.observe(span_type="function")
        def simple_function():
            return "result"

        result = simple_function()

        duration_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="trace",
            name="simple_function",
            response_time=duration_ms,
            response_length=len(result),
            exception=None,
        )

    @task(5)
    def trace_with_attributes(self):
        start = time.perf_counter()

        with self.tracer.span("attributed-span"):
            self.tracer.set_attribute("iteration", self.iteration)
            self.tracer.set_attribute("user_id", self.user_id)
            self.iteration += 1

        duration_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="trace",
            name="with_attributes",
            response_time=duration_ms,
            response_length=0,
            exception=None,
        )

    @task(3)
    def trace_with_tags(self):
        start = time.perf_counter()

        with self.tracer.span("tagged-span"):
            self.tracer.tag(["load-test", "performance"])

        duration_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="trace",
            name="with_tags",
            response_time=duration_ms,
            response_length=0,
            exception=None,
        )

    @task(2)
    def nested_spans(self):
        start = time.perf_counter()

        @self.tracer.observe(span_type="function")
        def outer():
            @self.tracer.observe(span_type="function")
            def inner():
                return "deep"

            return inner()

        result = outer()

        duration_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="trace",
            name="nested_spans",
            response_time=duration_ms,
            response_length=len(result),
            exception=None,
        )

    @task(1)
    def large_payload(self):
        start = time.perf_counter()

        @self.tracer.observe(span_type="function")
        def large_function():
            return "x" * 10000

        result = large_function()

        duration_ms = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="trace",
            name="large_payload",
            response_time=duration_ms,
            response_length=len(result),
            exception=None,
        )
