"""Unit tests for benchmarks/aggregate.py (written by Carleano)."""
import json
import sys

import pytest

from benchmarks.aggregate import (
    HEADER,
    build_rows,
    find_scenarios,
    fmt,
    load_runs,
    main,
    mean_of,
    render,
)


# ---------------------------------------------------------------- helpers

def make_group(n, ttft_p50, lat_p50):
    return {
        "num_requests": n,
        "output_tokens": n * 16,
        "ttft_mean": ttft_p50,
        "ttft": {"p50": ttft_p50, "p95": ttft_p50 * 2, "p99": ttft_p50 * 3},
        "latency_mean": lat_p50,
        "latency": {"p50": lat_p50, "p95": lat_p50 * 2, "p99": lat_p50 * 3},
    }


def make_summary(ttft_p50=0.5, lat_p50=2.0, tput=20.0, n=50, buckets=None):
    overall = make_group(n, ttft_p50, lat_p50)
    overall.update(
        {
            "failed_requests": 0,
            "duration_s": 100.0,
            "throughput_tokens_per_s": tput,
            "throughput_req_per_s": 0.5,
        }
    )
    return {"overall": overall, "by_bucket": buckets or {}}


def write_run(root, prefix, scenario, idx, summary):
    d = root / f"{prefix}_{scenario}_run{idx}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(summary))


def write_three_runs(root, prefix, scenario, buckets_fn=None):
    """Three runs with ttft_p50 = 0.4/0.5/0.6, lat_p50 = 2/4/6, tput = 10/20/30."""
    for i, (t, l, tp) in enumerate([(0.4, 2.0, 10.0), (0.5, 4.0, 20.0), (0.6, 6.0, 30.0)], 1):
        buckets = buckets_fn(t, l) if buckets_fn else None
        write_run(root, prefix, scenario, i, make_summary(t, l, tp, buckets=buckets))


# ------------------------------------------------------------------- fmt

def test_fmt_none_is_na():
    assert fmt(None) == "n/a"


def test_fmt_default_two_digits():
    assert fmt(1.2345) == "1.23"


def test_fmt_custom_digits():
    assert fmt(24.44, 1) == "24.4"


# -------------------------------------------------------- find_scenarios

def test_find_scenarios_groups_by_scenario(tmp_path):
    for name in ["fcfs_A_run1", "fcfs_A_run2", "fcfs_B_run1"]:
        (tmp_path / name).mkdir()
    assert find_scenarios(str(tmp_path), "fcfs") == ["A", "B"]


def test_find_scenarios_ignores_other_prefixes_and_dirs(tmp_path):
    for name in ["fcfs_A_run1", "load_fcfs_A_run1", "load_fcfs_B_run1", "warmup", "smoke"]:
        (tmp_path / name).mkdir()
    assert find_scenarios(str(tmp_path), "fcfs") == ["A"]
    assert find_scenarios(str(tmp_path), "load_fcfs") == ["A", "B"]


def test_find_scenarios_empty(tmp_path):
    assert find_scenarios(str(tmp_path), "fcfs") == []


# ------------------------------------------------------------- load_runs

def test_load_runs_returns_all_runs_of_scenario(tmp_path):
    write_three_runs(tmp_path, "fcfs", "A")
    runs = load_runs(str(tmp_path), "fcfs", "A")
    assert len(runs) == 3
    assert all("overall" in r and "by_bucket" in r for r in runs)


def test_load_runs_filters_scenario_and_prefix(tmp_path):
    write_run(tmp_path, "fcfs", "A", 1, make_summary())
    write_run(tmp_path, "fcfs", "B", 1, make_summary())
    write_run(tmp_path, "load_fcfs", "A", 1, make_summary())
    assert len(load_runs(str(tmp_path), "fcfs", "A")) == 1
    assert len(load_runs(str(tmp_path), "fcfs", "B")) == 1
    assert load_runs(str(tmp_path), "fcfs", "C") == []


# -------------------------------------------------------------- mean_of

def test_mean_of_nested_metric():
    runs = [make_summary(ttft_p50=t) for t in (0.4, 0.5, 0.6)]
    assert mean_of(runs, "overall", "ttft", "p50") == pytest.approx(0.5)


def test_mean_of_scalar_metric_with_key_none():
    runs = [make_summary(tput=t) for t in (10.0, 20.0, 30.0)]
    assert mean_of(runs, "overall", "throughput_tokens_per_s", None) == pytest.approx(20.0)


def test_mean_of_skips_runs_missing_the_group():
    with_long = make_summary(buckets={"long": make_group(13, 0.9, 15.0)})
    without_long = make_summary(buckets={})
    assert mean_of([with_long, without_long], "long", "ttft", "p50") == pytest.approx(0.9)


def test_mean_of_returns_none_when_group_never_present():
    runs = [make_summary(), make_summary()]
    assert mean_of(runs, "nonexistent", "ttft", "p50") is None


def test_mean_of_skips_none_values():
    a = make_summary(ttft_p50=0.4)
    b = make_summary(ttft_p50=0.8)
    b["overall"]["ttft"]["p50"] = None
    assert mean_of([a, b], "overall", "ttft", "p50") == pytest.approx(0.4)


# ------------------------------------------------------------ build_rows

def test_build_rows_single_group_scenario(tmp_path):
    write_three_runs(tmp_path, "fcfs", "A")
    rows = build_rows(str(tmp_path), "fcfs")
    assert len(rows) == 1
    scen, group, n_req, t50, t95, t99, l50, l95, l99, tput, runs = rows[0]
    assert (scen, group, n_req, runs) == ("A", "overall", "50", "3")
    # means over runs with ttft_p50 = 0.4 / 0.5 / 0.6
    assert (t50, t95, t99) == ("0.50", "1.00", "1.50")
    # means over runs with lat_p50 = 2 / 4 / 6
    assert (l50, l95, l99) == ("4.00", "8.00", "12.00")
    assert tput == "20.0"


def test_build_rows_with_buckets(tmp_path):
    def buckets(t, l):
        return {
            "short": make_group(37, t / 2, l / 2),
            "long": make_group(13, t * 2, l * 2),
        }

    write_three_runs(tmp_path, "fcfs", "B", buckets_fn=buckets)
    rows = build_rows(str(tmp_path), "fcfs")

    # overall first, then buckets in alphabetical order
    assert [r[1] for r in rows] == ["overall", "long", "short"]
    by_group = {r[1]: r for r in rows}
    assert by_group["short"][2] == "37"
    assert by_group["long"][2] == "13"
    # throughput is only reported for the overall row
    assert by_group["overall"][9] == "20.0"
    assert by_group["short"][9] == "n/a"
    assert by_group["long"][9] == "n/a"
    # short ttft p50 = mean(0.2, 0.25, 0.3) = 0.25
    assert by_group["short"][3] == "0.25"
    assert by_group["long"][3] == "1.00"


def test_build_rows_multiple_scenarios_sorted(tmp_path):
    write_three_runs(tmp_path, "fcfs", "B")
    write_three_runs(tmp_path, "fcfs", "A")
    rows = build_rows(str(tmp_path), "fcfs")
    assert [r[0] for r in rows] == ["A", "B"]


def test_build_rows_isolates_prefixes(tmp_path):
    write_three_runs(tmp_path, "fcfs", "A")
    write_three_runs(tmp_path, "load_fcfs", "A")
    write_run(tmp_path, "load_fcfs", "B", 1, make_summary())
    assert len(build_rows(str(tmp_path), "fcfs")) == 1
    assert {r[0] for r in build_rows(str(tmp_path), "load_fcfs")} == {"A", "B"}


def test_build_rows_no_results(tmp_path):
    assert build_rows(str(tmp_path), "fcfs") == []


def test_build_rows_row_length_matches_header(tmp_path):
    write_three_runs(tmp_path, "fcfs", "A")
    for row in build_rows(str(tmp_path), "fcfs"):
        assert len(row) == len(HEADER)


# --------------------------------------------------------------- render

def test_render_produces_aligned_ascii_table(tmp_path):
    write_three_runs(tmp_path, "fcfs", "A")
    text = render(build_rows(str(tmp_path), "fcfs"))
    lines = text.splitlines()
    assert len({len(line) for line in lines}) == 1  # every line same width
    assert lines[0].startswith("+-") and lines[0].endswith("-+")
    assert "ttft_p50" in lines[1]
    assert lines[2] == lines[0]  # separator under the header
    assert lines[-1] == lines[0]
    assert text.isascii()  # midterm email must be plain ASCII


def test_render_header_only_when_no_rows():
    text = render([])
    assert "scen" in text and "runs" in text


# ----------------------------------------------------------------- main

def test_main_prints_table(tmp_path, monkeypatch, capsys):
    write_three_runs(tmp_path, "fcfs", "A")
    monkeypatch.setattr(sys, "argv", ["aggregate", str(tmp_path), "fcfs"])
    main()
    out = capsys.readouterr().out
    assert "All times in seconds" in out
    assert "ttft_p50" in out
    assert "overall" in out


def test_main_reports_when_nothing_found(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aggregate", str(tmp_path), "fcfs"])
    main()
    assert "no results found" in capsys.readouterr().out