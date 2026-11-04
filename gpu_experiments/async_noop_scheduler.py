"""Async scheduler control for vLLM 0.26.0."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class AsyncNoopScheduler(AsyncScheduler):
    """Preserve the native asynchronous scheduling behavior."""

    pass
