"""Length-aware scheduling with native FCFS queue selection."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class LengthAwareQueueOnlyScheduler(AsyncScheduler):
    """Sort waiting queues by estimated request length."""

    @staticmethod
    def _length_key(request):
        return (
            request.num_prompt_tokens + request.max_tokens,
            request.arrival_time,
        )

    def _sort_queue(self, queue):
        if len(queue) > 1:
            requests = sorted(queue, key=self._length_key)
            queue.clear()
            queue.extend(requests)

    def schedule(self, throttle_prefills=False):
        # Prioritize shorter requests within each queue.
        self._sort_queue(self.waiting)
        self._sort_queue(self.skipped_waiting)

        # Preserve the native FCFS queue-selection behavior.
        return super().schedule(throttle_prefills)
