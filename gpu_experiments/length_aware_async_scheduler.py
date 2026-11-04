"""Length-aware asynchronous scheduling for vLLM 0.26.0."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class LengthAwareAsyncScheduler(AsyncScheduler):
    """Prioritize shorter estimated requests in waiting queues."""

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
        # Reorder waiting queues without touching running requests.
        self._sort_queue(self.waiting)
        self._sort_queue(self.skipped_waiting)

        # Retain AsyncScheduler's inherited scheduling behavior.
        return super().schedule(throttle_prefills)

    def _select_waiting_queue_for_scheduling(self):
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()

            if self._length_key(waiting_req) <= self._length_key(skipped_req):
                return self.waiting

            return self.skipped_waiting

        return self.waiting or self.skipped_waiting or None
