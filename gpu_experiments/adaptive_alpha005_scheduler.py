from gpu_experiments.adaptive_length_aware_scheduler import AdaptiveLengthAwareScheduler

class AdaptiveAlpha005Scheduler(AdaptiveLengthAwareScheduler):
    AGING_ALPHA = 0.05
