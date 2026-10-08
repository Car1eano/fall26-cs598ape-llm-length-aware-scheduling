"""Synthetic request mixes: (weight, bucket, approximate prompt tokens, max output tokens)."""
SCENARIOS = {
    "A_uniform": [(1.0, "medium", 64, 64)],
    "B_mixed": [(0.8, "short", 32, 16), (0.2, "long", 256, 128)],
}
