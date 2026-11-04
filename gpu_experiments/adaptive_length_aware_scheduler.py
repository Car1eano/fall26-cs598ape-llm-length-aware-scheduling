"""Length-aware scheduling with smooth waiting-time aging."""

import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class AdaptiveLengthAwareScheduler(AsyncScheduler):
    """Prioritize short requests while gradually aging waiting requests."""

    AGING_ALPHA = 0.1

    def _priority_key(self, request, now):
        # Estimate the request's total token length.
        estimated_length = (
            request.num_prompt_tokens + request.max_tokens
        )

        # Measure elapsed time since request arrival.
        waiting_time = max(0.0, now - request.arrival_time)

        # Gradually reduce the priority score as time increases.
        priority = estimated_length / (
            1.0 + self.AGING_ALPHA * waiting_time
        )

        # Lower scores are scheduled first.
        return (priority, request.arrival_time)

    def _sort_queue(self, queue, now):
        # A queue with zero or one request needs no sorting.
        if len(queue) <= 1:
            return

        # Sort requests using the aging-adjusted priority.
        requests = sorted(
            queue,
            key=lambda request: self._priority_key(request, now),
        )

        # Replace the queue contents with the sorted requests.
        queue.clear()
        queue.extend(requests)

    def schedule(self, throttle_prefills=False):
        # Use one timestamp for both queues.
        now = time.time()

        # Reorder requests inside each waiting queue.
        self._sort_queue(self.waiting, now)
        self._sort_queue(self.skipped_waiting, now)

        # Preserve native FCFS cross-queue selection.
        return super().schedule(throttle_prefills)
