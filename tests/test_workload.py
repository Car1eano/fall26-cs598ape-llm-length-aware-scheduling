"""Tests for deterministic arrival generation and scenario distributions."""
import pytest
from workload.generator import generate_workload, read_jsonl, write_jsonl


def test_count_and_schema():
    rows = generate_workload("B_mixed", 50, 2.0, 1)
    assert len(rows) == 50
    assert all(set(row) == {"id", "arrival_time_s", "prompt", "max_tokens", "bucket"} for row in rows)


def test_arrivals_increase():
    times = [r["arrival_time_s"] for r in generate_workload("A_uniform", 100, 2.0)]
    assert times[0] > 0
    assert all(a < b for a, b in zip(times, times[1:]))


def test_seed_reproducibility():
    assert generate_workload("B_mixed", 30, 1.0, 42) == generate_workload("B_mixed", 30, 1.0, 42)
    assert generate_workload("B_mixed", 30, 1.0, 1) != generate_workload("B_mixed", 30, 1.0, 2)


def test_mixed_distribution():
    rows = generate_workload("B_mixed", 2000, 1.0, 0)
    short = sum(r["bucket"] == "short" for r in rows)
    assert 0.75 < short / len(rows) < 0.85
    assert all(r["max_tokens"] == (16 if r["bucket"] == "short" else 128) for r in rows)


def test_rate():
    rows = generate_workload("A_uniform", 3000, 4.0, 0)
    assert len(rows) / rows[-1]["arrival_time_s"] == pytest.approx(4.0, rel=0.1)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        generate_workload("unknown", 10, 1.0)
    with pytest.raises(ValueError):
        generate_workload("A_uniform", 10, 0)
    with pytest.raises(ValueError):
        generate_workload("A_uniform", -1, 1.0)


def test_jsonl_roundtrip(tmp_path):
    rows = generate_workload("A_uniform", 3, 1.0)
    path = tmp_path / "nested" / "workload.jsonl"
    write_jsonl(rows, path)
    assert read_jsonl(path) == rows
