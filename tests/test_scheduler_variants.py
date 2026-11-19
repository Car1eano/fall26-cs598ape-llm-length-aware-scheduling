"""Check the priority rules added for the scheduler sensitivity study."""

from types import SimpleNamespace

import pytest

from gpu_experiments.scheduler_variants import (
    AgingAlpha1, AgingAlpha2, AgingAlpha5, AgingAlpha10,
    HRRNScheduler, NoisyLengthScheduler, PromptOnlyScheduler, WaitPromotionScheduler,
)


def request(prompt, output, arrival, scale=1.0):
    return SimpleNamespace(
        num_prompt_tokens=prompt, max_tokens=output, arrival_time=arrival,
        sampling_params=SimpleNamespace(extra_args={"length_noise_scale": scale}),
    )


def test_prompt_only_ignores_max_tokens():
    short_prompt = request(32, 128, 2)
    long_prompt = request(64, 16, 1)
    assert PromptOnlyScheduler._length_key(short_prompt) < PromptOnlyScheduler._length_key(long_prompt)


def test_noise_can_reverse_similar_estimates():
    shorter = request(32, 16, 1, scale=1.3)
    longer = request(40, 20, 2, scale=0.7)
    assert NoisyLengthScheduler._length_key(longer) < NoisyLengthScheduler._length_key(shorter)


@pytest.mark.parametrize(("scheduler_type", "alpha"), [
    (AgingAlpha1, 1), (AgingAlpha2, 2), (AgingAlpha5, 5), (AgingAlpha10, 10),
])
def test_aging_alpha(scheduler_type, alpha):
    scheduler = scheduler_type.__new__(scheduler_type)
    score, _ = scheduler._priority_key(request(32, 16, 1), now=3)
    assert score == pytest.approx(48 / (1 + alpha * 2))


def test_wait_promotion_has_threshold_and_fifo_tie_break():
    scheduler = WaitPromotionScheduler.__new__(WaitPromotionScheduler)
    older = request(512, 128, 0)
    long = request(256, 128, 1)
    recent_short = request(32, 16, 9)
    assert scheduler._priority_key(recent_short, 10) < scheduler._priority_key(long, 10)
    assert scheduler._priority_key(older, 11) < scheduler._priority_key(long, 11)
    assert scheduler._priority_key(long, 11) < scheduler._priority_key(recent_short, 11)


def test_hrrn_uses_wait_relative_to_service_cost():
    scheduler = HRRNScheduler.__new__(HRRNScheduler)
    long = request(256, 128, 0)
    short = request(32, 16, 17)
    assert scheduler._priority_key(short, 20) < scheduler._priority_key(long, 20)
    assert scheduler._priority_key(long, 18) < scheduler._priority_key(short, 18)
