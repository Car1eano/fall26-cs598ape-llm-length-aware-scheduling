"""Instrumented smooth-aging scheduler for diagnostic experiments."""

import logging
import time

from gpu_experiments.adaptive_alpha05_scheduler import AdaptiveAlpha05Scheduler


logger = logging.getLogger(__name__)


class InstrumentedAlpha05Scheduler(AdaptiveAlpha05Scheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instrumented_sort_calls = 0
        self.instrumented_changed_calls = 0
        self.instrumented_changed_pairs = 0
        self.instrumented_long_over_short = 0

    @staticmethod
    def _length_key(request):
        return (
            request.num_prompt_tokens + request.max_tokens,
            request.arrival_time,
        )

    @staticmethod
    def _estimated_length(request):
        return request.num_prompt_tokens + request.max_tokens

    def _sort_queue(self, queue, now):
        if len(queue) <= 1:
            return super()._sort_queue(queue, now)

        self.instrumented_sort_calls += 1

        baseline = sorted(queue, key=self._length_key)
        aged = sorted(queue, key=lambda r: self._priority_key(r, now))

        baseline_ids = [r.request_id for r in baseline]
        aged_ids = [r.request_id for r in aged]

        if baseline_ids != aged_ids:
            self.instrumented_changed_calls += 1

            baseline_positions = {
                r.request_id: i for i, r in enumerate(baseline)
            }

            changed_pairs = 0
            long_over_short = 0

            for i, first in enumerate(aged):
                for second in aged[i + 1:]:
                    if (
                        baseline_positions[first.request_id]
                        > baseline_positions[second.request_id]
                    ):
                        changed_pairs += 1

                        if (
                            self._estimated_length(first)
                            > self._estimated_length(second)
                        ):
                            long_over_short += 1

            self.instrumented_changed_pairs += changed_pairs
            self.instrumented_long_over_short += long_over_short

            logger.warning(
                "AGING_REORDER changed_calls=%d "
                "changed_pairs=%d long_over_short=%d "
                "total_changed_pairs=%d "
                "total_long_over_short=%d",
                self.instrumented_changed_calls,
                changed_pairs,
                long_over_short,
                self.instrumented_changed_pairs,
                self.instrumented_long_over_short,
            )

        return super()._sort_queue(queue, now)
