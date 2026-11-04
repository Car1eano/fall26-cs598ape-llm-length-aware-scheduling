"""Length-aware scheduling for vLLM 0.26.0."""

from vllm.v1.core.sched.scheduler import Scheduler


class LengthAwareScheduler(Scheduler):
    """Prioritize requests with shorter estimated lengths."""

    @staticmethod
    def _length_key(request):
        # Estimate total work using prompt and maximum output length.
        estimated_length = (
            request.num_prompt_tokens + request.max_tokens
        )

        # Earlier arrivals win ties.
        return estimated_length, request.arrival_time

    def schedule(self, throttle_prefills=False):
        # Sort schedulable waiting requests.
        if self.waiting:
            requests = sorted(
                self.waiting,
                key=self._length_key,
            )

            self.waiting.clear()
            self.waiting.extend(requests)

        # Preserve vLLM's running-request and KV-cache logic.
        return super().schedule(throttle_prefills)

    def _select_waiting_queue_for_scheduling(self):
        # If both queues have requests, compare their heads.
        if self.waiting and self.skipped_waiting:
            waiting_request = self.waiting.peek_request()
            skipped_request = self.skipped_waiting.peek_request()

            if self._length_key(waiting_request) <= self._length_key(
                skipped_request
            ):
                return self.waiting

            return self.skipped_waiting

        # Otherwise, return whichever queue contains requests.
        return self.waiting or self.skipped_waiting or None
