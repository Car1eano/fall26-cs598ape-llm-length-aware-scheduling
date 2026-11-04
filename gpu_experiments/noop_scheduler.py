"""Control scheduler with unchanged vLLM behavior."""

from vllm.v1.core.sched.scheduler import Scheduler


class NoopScheduler(Scheduler):
    """Inherit the default scheduler without overriding methods."""
    pass
