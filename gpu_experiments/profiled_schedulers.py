"""Measure scheduler CPU cost and scheduled batch composition in vLLM."""

import json
import os
from pathlib import Path
from time import perf_counter_ns

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from gpu_experiments.adaptive_alpha05_scheduler import AdaptiveAlpha05Scheduler
from gpu_experiments.length_aware_queue_only_scheduler import LengthAwareQueueOnlyScheduler

class SchedulerProfile:
    """Emit cumulative counters every 128 nonempty engine steps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_path = Path(os.environ["CS598_SCHED_PROFILE_PATH"])
        self.profile_interval = int(os.environ.get("CS598_SCHED_PROFILE_INTERVAL", "128"))
        self.profile = dict(steps=0, schedule_calls=0, schedule_ns=0, sort_ns=0, sort_calls=0,
                            max_waiting=0, deep_queue_steps=0, deep_queue_schedule_ns=0,
                            deep_queue_sort_ns=0, slots=0, prefill_tokens=0, decode_tokens=0)

    def _sort_queue(self, queue, *args):
        start = perf_counter_ns()
        super()._sort_queue(queue, *args)
        self.profile["sort_ns"] += perf_counter_ns() - start
        self.profile["sort_calls"] += 1

    def schedule(self, *args, **kwargs):
        waiting = len(self.waiting) + len(self.skipped_waiting)
        sort_before = self.profile["sort_ns"]
        start = perf_counter_ns()
        output = super().schedule(*args, **kwargs)
        elapsed = perf_counter_ns() - start
        scheduled = output.num_scheduled_tokens
        prefill = 0
        for request_id, tokens in scheduled.items():
            request = self.requests[request_id]
            remaining_prefill = request.num_prompt_tokens - (request.num_computed_tokens - tokens)
            prefill += max(0, min(remaining_prefill, tokens))
        self.profile["schedule_ns"] += elapsed
        self.profile["schedule_calls"] += 1
        self.profile["max_waiting"] = max(waiting, self.profile["max_waiting"])
        if waiting >= 100:
            self.profile["deep_queue_steps"] += 1
            self.profile["deep_queue_schedule_ns"] += elapsed
            self.profile["deep_queue_sort_ns"] += self.profile["sort_ns"] - sort_before
        if scheduled:
            self.profile["steps"] += 1
            self.profile["slots"] += len(scheduled)
            self.profile["prefill_tokens"] += prefill
            self.profile["decode_tokens"] += sum(scheduled.values()) - prefill
            if self.profile["steps"] % self.profile_interval == 0:
                with self.profile_path.open("a") as output_file:
                    output_file.write(json.dumps(self.profile, sort_keys=True) + "\n")
        return output


class ProfiledFCFS(SchedulerProfile, AsyncScheduler):
    pass


class ProfiledQueueOnly(SchedulerProfile, LengthAwareQueueOnlyScheduler):
    pass


class ProfiledAging(SchedulerProfile, AdaptiveAlpha05Scheduler):
    pass


class AvoidUnchangedAgingQueue(AdaptiveAlpha05Scheduler):
    """Retain the exact stable sort and avoid rewriting an unchanged queue."""

    def _sort_queue(self, queue, now):
        if len(queue) <= 1:
            return
        original = list(queue)
        ordered = sorted(original, key=lambda request: self._priority_key(request, now))
        if any(before is not after for before, after in zip(original, ordered)):
            queue.clear()
            queue.extend(ordered)


class ProfiledOptimizedAging(SchedulerProfile, AvoidUnchangedAgingQueue):
    pass
