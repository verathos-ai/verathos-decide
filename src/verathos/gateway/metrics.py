"""Prometheus metrics of one gateway process (served on ``metrics_port``, 127.0.0.1)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)

REQUESTS = Counter("gateway_requests_total", "User requests by model and outcome", ["model", "outcome"])
REQUEST_SECONDS = Histogram("gateway_request_seconds", "User request latency, retries included", ["model"],
                            buckets=LATENCY_BUCKETS)
UPSTREAM_SECONDS = Histogram("gateway_upstream_seconds", "Latency of answered attempts at miners", ["model"],
                             buckets=LATENCY_BUCKETS)
UPSTREAM_FAILURES = Counter("gateway_upstream_failures_total", "Failed attempts at miners by kind", ["kind"])
RETRIES = Counter("gateway_retries_total", "Attempts after a failed one")
IN_FLIGHT = Gauge("gateway_in_flight", "User requests in flight")
ROUTES = Gauge("gateway_routes", "Routable endpoints by model", ["model"])
ROUTES_AGE = Gauge("gateway_routes_age_seconds", "Age of the routes in use")
BREAKERS_OPEN = Gauge("gateway_breakers_open", "Endpoints whose circuit breaker is open")
USAGE_FLUSH_FAILURES = Counter("gateway_usage_flush_failures_total", "Failed usage writes")
SAMPLES = Counter("gateway_samples_total", "Sampled exchanges by outcome (submitted, kept, failed)", ["outcome"])
IMAGES = Counter("gateway_images_total", "Users' images by outcome (canonical, refused, failed: the decoder failed)",
                 ["outcome"])
IMAGE_SECONDS = Histogram("gateway_image_seconds", "Time to make one image canonical, in a decoder",
                          buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0))
IMAGE_BYTES = Histogram("gateway_image_bytes", "Bytes of the images made canonical",
                        buckets=(16 << 10, 64 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20))
