"""Experimental waiting-queue policies for the P1 sensitivity study."""

import time

from gpu_experiments.adaptive_length_aware_scheduler import AdaptiveLengthAwareScheduler
from gpu_experiments.length_aware_queue_only_scheduler import LengthAwareQueueOnlyScheduler


class PromptOnlyScheduler(LengthAwareQueueOnlyScheduler):
    @staticmethod
    def _length_key(request):
        return request.num_prompt_tokens, request.arrival_time


class NoisyLengthScheduler(LengthAwareQueueOnlyScheduler):
    @staticmethod
    def _length_key(request):
        extra_args = request.sampling_params.extra_args or {}
        scale = extra_args.get("length_noise_scale", 1.0)
        length = request.num_prompt_tokens + request.max_tokens
        return length * scale, request.arrival_time


class AgingAlpha1(AdaptiveLengthAwareScheduler):
    AGING_ALPHA = 1.0


class AgingAlpha2(AdaptiveLengthAwareScheduler):
    AGING_ALPHA = 2.0


class AgingAlpha5(AdaptiveLengthAwareScheduler):
    AGING_ALPHA = 5.0


class AgingAlpha10(AdaptiveLengthAwareScheduler):
    AGING_ALPHA = 10.0


class WaitPromotionScheduler(AdaptiveLengthAwareScheduler):
    """Prioritize requests within a queue after ten seconds of waiting."""

    WAIT_THRESHOLD_S = 10.0

    def _priority_key(self, request, now):
        if now - request.arrival_time >= self.WAIT_THRESHOLD_S:
            return 0, request.arrival_time, 0
        return 1, request.num_prompt_tokens + request.max_tokens, request.arrival_time


class WaitLimit5Scheduler(WaitPromotionScheduler):
    """Promote overdue heads across queues at the next eligible admission."""

    WAIT_THRESHOLD_S = 5.0

    def _select_waiting_queue_for_scheduling(self):
        now = time.time()
        overdue = [
            queue for queue in (self.waiting, self.skipped_waiting)
            if queue and now - queue.peek_request().arrival_time >= self.WAIT_THRESHOLD_S
        ]
        if overdue:
            return min(overdue, key=lambda queue: queue.peek_request().arrival_time)
        return super()._select_waiting_queue_for_scheduling()


class HRRNScheduler(AdaptiveLengthAwareScheduler):
    """Order by decreasing response ratio, with length as a service-cost proxy."""

    def _priority_key(self, request, now):
        estimated_size = max(1, request.num_prompt_tokens + request.max_tokens)
        waiting = max(0.0, now - request.arrival_time)
        # A common conversion factor from tokens to seconds preserves this order.
        return -waiting / estimated_size, request.arrival_time
