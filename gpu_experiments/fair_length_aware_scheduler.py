"""Fair length-aware asynchronous scheduling for vLLM 0.26.0."""

import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class FairLengthAwareScheduler(AsyncScheduler):
    """Prioritize short requests while aging older waiting requests."""

    MAX_WAIT_SECONDS = 2.0

    def _fairness_key(self, request, now):
        # Calculate the elapsed time since the request arrived.
        waiting_time = max(0.0, now - request.arrival_time)

        # Give aged requests higher priority.
        if waiting_time >= self.MAX_WAIT_SECONDS:
            return (
                0,
                request.arrival_time,
            )

        # Otherwise, prioritize shorter estimated requests.
        return (
            1,
            request.num_prompt_tokens + request.max_tokens,
            request.arrival_time,
        )

    def _sort_queue(self, queue, now):
        # Avoid unnecessary sorting for zero or one request.
        if len(queue) <= 1:
            return

        # Sort requests using aging and estimated length.
        requests = sorted(
            queue,
            key=lambda request: self._fairness_key(request, now),
        )

        # Rebuild the queue in priority order.
        queue.clear()
        queue.extend(requests)

    def schedule(self, throttle_prefills=False):
        # Use the same clock as Request.arrival_time.
        now = time.time()

        # Apply fairness within each waiting queue.
        self._sort_queue(self.waiting, now)
        self._sort_queue(self.skipped_waiting, now)

        # Preserve native AsyncScheduler queue selection.
        return super().schedule(throttle_prefills)
