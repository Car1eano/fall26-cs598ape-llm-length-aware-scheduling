"""Summarize successful streaming responses and failed request counts."""
import numpy as np


def _percentiles(values):
    return {f"p{p}": float(np.percentile(values, p)) if values else None
            for p in (50, 95, 99)}


def _group_stats(rows):
    ttft = [r["first_token_time"] - r["send_time"] for r in rows]
    latency = [r["finish_time"] - r["send_time"] for r in rows]
    return {"num_requests": len(rows),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in rows),
            "ttft_mean": float(np.mean(ttft)) if ttft else None,
            "ttft": _percentiles(ttft),
            "latency_mean": float(np.mean(latency)) if latency else None,
            "latency": _percentiles(latency)}


def summarize(rows):
    ok = [r for r in rows if r.get("error") is None
          and r.get("first_token_time") is not None]
    duration = (max(r["finish_time"] for r in ok) -
                min(r["send_time"] for r in ok)) if ok else 0.0
    overall = _group_stats(ok)
    overall.update({"failed_requests": len(rows) - len(ok),
                    "duration_s": duration,
                    "throughput_tokens_per_s": overall["output_tokens"] / duration if duration > 0 else 0.0,
                    "throughput_req_per_s": len(ok) / duration if duration > 0 else 0.0})
    by_bucket = {bucket: _group_stats([r for r in ok if r["bucket"] == bucket])
                 for bucket in sorted({r["bucket"] for r in ok})}
    return {"overall": overall, "by_bucket": by_bucket}
