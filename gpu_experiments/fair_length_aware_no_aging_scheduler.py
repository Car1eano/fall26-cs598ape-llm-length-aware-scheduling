"""Control scheduler with aging effectively disabled."""

from gpu_experiments.fair_length_aware_scheduler import FairLengthAwareScheduler


class FairLengthAwareNoAgingScheduler(FairLengthAwareScheduler):
    """Keep the same scheduler logic without triggering aging."""

    MAX_WAIT_SECONDS = 1000000000.0
