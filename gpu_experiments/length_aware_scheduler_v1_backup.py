"""Length-aware waiting queue scheduling for vLLM 0.26.0."""

from vllm.v1.core.sched.scheduler import Scheduler


class LengthAwareScheduler(Scheduler):
    """Prioritize shorter estimated requests in the waiting queue."""

    def schedule(self, throttle_prefills=False):
        # Reorder only the waiting requests.
        # Do not modify running requests or KV cache allocation.

        if self.waiting:
            requests = list(self.waiting)

            # Smaller estimated total length gets higher priority.
            # Arrival time breaks ties deterministically.
            requests.sort(
                key=lambda request: (
                    request.num_prompt_tokens + request.max_tokens,
                    request.arrival_time,
                )
            )

            # Preserve the existing FCFS queue object.
            self.waiting.clear()
            self.waiting.extend(requests)

        # Keep vLLM's original scheduling logic unchanged.
        return super().schedule(throttle_prefills)
