"""Async streaming benchmark client for vLLM's /v1/completions endpoint."""
import asyncio
import json
import time
import aiohttp


async def send_one(session, url, model, req, t0):
    # Schedule by time relative to the start of this benchmark run.
    delay = req["arrival_time_s"] - (time.perf_counter() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    send_time = time.perf_counter() - t0
    payload = {"model": model, "prompt": req["prompt"],
               "max_tokens": req["max_tokens"], "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True},
               "ignore_eos": True}
    first_token_time = None
    output_tokens = 0
    usage_tokens = None
    error = None
    try:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            # SSE frames can cross arbitrary TCP chunk boundaries.
            buffer = ""
            async for chunk in response.content.iter_any():
                buffer += chunk.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    event = json.loads(data)
                    if event.get("usage"):
                        usage_tokens = event["usage"].get("completion_tokens")
                    for choice in event.get("choices") or []:
                        if choice.get("text"):
                            output_tokens += 1  # Fallback: text chunks, not exact tokens.
                            if first_token_time is None:
                                first_token_time = time.perf_counter() - t0
    except Exception as exc:
        error = str(exc)
    finish_time = time.perf_counter() - t0
    return {"id": req["id"], "bucket": req["bucket"],
            "send_time": send_time, "first_token_time": first_token_time,
            "finish_time": finish_time,
            "output_tokens": usage_tokens if usage_tokens is not None else output_tokens,
            "error": error}


async def run_workload(requests, url, model):
    t0 = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        return await asyncio.gather(*(send_one(session, url, model, req, t0)
                                      for req in requests))
