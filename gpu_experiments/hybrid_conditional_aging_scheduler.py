"""Length-aware scheduling with conditional aging."""

import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class HybridConditionalAgingScheduler(AsyncScheduler):
    """Use length priority first, then activate aging after a delay."""

    AGING_ALPHA = 0.5
    AGING_DELAY_SECONDS = 5.0

    @classmethod
    def _priority_key(cls, request, now):
        estimated_length = (
            request.num_prompt_tokens + request.max_tokens
        )

        age = max(0.0, now - request.arrival_time)

        effective_age = max(
            0.0, age - cls.AGING_DELAY_SECONDS
        )

        priority = estimated_length / (
            1.0 + cls.AGING_ALPHA * effective_age
        )

        return (priority, request.arrival_time)

    def _sort_queue(self, queue, now):
        if len(queue) <= 1:
            return

        requests = sorted(
            queue,
            key=lambda request: self._priority_key(
                request, now
            ),
        )

        queue.clear()
        queue.extend(requests)

    def schedule(self, throttle_prefills=False):
        now = time.time()

        self._sort_queue(self.waiting, now)
        self._sort_queue(self.skipped_waiting, now)

        return super().schedule(throttle_prefills)
