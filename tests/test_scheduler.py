"""Unit tests for custom vLLM queue scheduling policies."""

from collections import deque
from types import SimpleNamespace

import pytest

from gpu_experiments.length_aware_queue_only_scheduler import (
    LengthAwareQueueOnlyScheduler,
)
from gpu_experiments.adaptive_alpha05_scheduler import (
    AdaptiveAlpha05Scheduler,
)


def request(prompt, output, arrival):
    """Create a lightweight request for queue-ordering tests."""
    return SimpleNamespace(
        num_prompt_tokens=prompt,
        max_tokens=output,
        arrival_time=arrival,
    )


def test_queue_only_shortest_first():
    scheduler = LengthAwareQueueOnlyScheduler.__new__(
        LengthAwareQueueOnlyScheduler
    )

    long_req = request(256, 128, 1.0)
    short_req = request(32, 16, 2.0)
    medium_req = request(128, 64, 3.0)

    queue = deque([long_req, medium_req, short_req])

    scheduler._sort_queue(queue)

    assert list(queue) == [short_req, medium_req, long_req]


def test_queue_only_tie_break_by_arrival():
    scheduler = LengthAwareQueueOnlyScheduler.__new__(
        LengthAwareQueueOnlyScheduler
    )

    early = request(32, 16, 1.0)
    late = request(16, 32, 2.0)

    queue = deque([late, early])
    scheduler._sort_queue(queue)

    assert list(queue) == [early, late]


@pytest.mark.parametrize("count", [0, 1])
def test_queue_only_empty_or_single(count):
    scheduler = LengthAwareQueueOnlyScheduler.__new__(
        LengthAwareQueueOnlyScheduler
    )

    queue = deque([request(32, 16, 1.0)] * count)
    original = list(queue)

    scheduler._sort_queue(queue)

    assert list(queue) == original


def test_aging_alpha_is_half():
    assert AdaptiveAlpha05Scheduler.AGING_ALPHA == 0.5


def test_aging_priority_formula():
    scheduler = AdaptiveAlpha05Scheduler.__new__(
        AdaptiveAlpha05Scheduler
    )

    req = request(256, 128, 10.0)

    score = scheduler._priority_key(req, now=14.0)

    expected = 384 / (1 + 0.5 * 4)

    assert score[0] == pytest.approx(expected)


def test_aging_priority_decreases_with_wait():
    scheduler = AdaptiveAlpha05Scheduler.__new__(
        AdaptiveAlpha05Scheduler
    )

    req = request(256, 128, 10.0)

    early_score = scheduler._priority_key(req, now=11.0)[0]
    later_score = scheduler._priority_key(req, now=20.0)[0]

    assert later_score < early_score


def test_aging_can_prioritize_waiting_long_request():
    scheduler = AdaptiveAlpha05Scheduler.__new__(
        AdaptiveAlpha05Scheduler
    )

    long_req = request(256, 128, 0.0)
    short_req = request(32, 16, 20.0)

    queue = deque([short_req, long_req])

    scheduler._sort_queue(queue, now=20.0)

    assert list(queue) == [long_req, short_req]


def test_aging_tie_break_by_arrival():
    scheduler = AdaptiveAlpha05Scheduler.__new__(
        AdaptiveAlpha05Scheduler
    )

    early = request(32, 16, 1.0)
    late = request(32, 16, 1.0)

    queue = deque([late, early])

    scheduler._sort_queue(queue, now=1.0)

    assert list(queue) == [late, early]


@pytest.mark.parametrize("count", [0, 1])
def test_aging_empty_or_single(count):
    scheduler = AdaptiveAlpha05Scheduler.__new__(
        AdaptiveAlpha05Scheduler
    )

    queue = deque([request(32, 16, 1.0)] * count)
    original = list(queue)

    scheduler._sort_queue(queue, now=10.0)

    assert list(queue) == original
