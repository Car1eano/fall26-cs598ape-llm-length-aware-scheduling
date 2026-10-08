"""Tests for TTFT, latency percentiles, throughput, and error handling."""
import pytest
from benchmarks.metrics import summarize


def make_row(i, bucket="short", ttft=1.0, latency=2.0, tokens=10, error=None):
    return {"id": i, "bucket": bucket, "send_time": 0.0,
            "first_token_time": ttft if error is None else None,
            "finish_time": latency, "output_tokens": tokens, "error": error}


def test_percentiles():
    rows = [make_row(i, ttft=float(i), latency=float(i + 1)) for i in range(1, 101)]
    stats = summarize(rows)["overall"]
    assert stats["ttft"]["p50"] == pytest.approx(50.5)
    assert stats["ttft"]["p95"] == pytest.approx(95.05)
    assert stats["ttft"]["p99"] == pytest.approx(99.01)
    assert stats["latency"]["p50"] == pytest.approx(51.5)


def test_throughput():
    rows = [make_row(i, latency=float(i + 1)) for i in range(1, 101)]
    stats = summarize(rows)["overall"]
    assert stats["duration_s"] == pytest.approx(101)
    assert stats["throughput_tokens_per_s"] == pytest.approx(1000 / 101)
    assert stats["throughput_req_per_s"] == pytest.approx(100 / 101)


def test_failed_requests_excluded():
    rows = [make_row(0), make_row(1), make_row(2, error="boom")]
    stats = summarize(rows)["overall"]
    assert stats["num_requests"] == 2
    assert stats["failed_requests"] == 1


def test_buckets():
    rows = [make_row(0), make_row(1), make_row(2, bucket="long", ttft=5, latency=9)]
    stats = summarize(rows)["by_bucket"]
    assert stats["short"]["num_requests"] == 2
    assert stats["long"]["ttft"]["p50"] == pytest.approx(5)


def test_empty():
    stats = summarize([])["overall"]
    assert stats["num_requests"] == 0
    assert stats["throughput_tokens_per_s"] == 0
    assert stats["ttft"]["p50"] is None
