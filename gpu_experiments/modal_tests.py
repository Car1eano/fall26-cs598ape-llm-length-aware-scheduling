"""Run the complete pytest suite against the production vLLM image."""

import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from uuid import uuid4

import modal

from gpu_experiments.modal_smoke import REMOTE_ROOT, ROOT, base_image, with_repo_sources

app = modal.App("cs598ape-vllm-tests")
test_image = with_repo_sources(base_image.pip_install("pytest==9.0.2"),
                               ("benchmarks", "workload", "gpu_experiments", "tests"))


@app.function(image=test_image, cpu=2, memory=4096, timeout=180)
def pytest_in_image():
    metadata = (
        f"Executed at (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"Installed vLLM: {version('vllm')}\n"
        "Command: python -m pytest tests/ -v\n"
    )
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v"], cwd=REMOTE_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=150,
    )
    return result.returncode, metadata + "\n" + result.stdout


@app.local_entrypoint()
def main():
    print("CPU-only test batch: estimated compute cost below US$0.01", flush=True)
    code, results = pytest_in_image.remote()
    directory = ROOT / "results/modal_validation"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"pytest-{uuid4().hex}.txt"
    path.write_text(results)
    print(results)
    print(f"Saved {path}")
    if code:
        raise SystemExit(code)
