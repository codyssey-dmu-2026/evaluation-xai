"""Runtime fingerprint and crash-safe file publication for real replays.

PPO policy outputs differ at float32 level between platforms/BLAS builds, so a
replay result is only traceable together with the runtime that produced it.
"""

import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def runtime_info():
    """Versions that can change PPO float32 arithmetic; no secrets or user paths."""
    info = {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "threads_env": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }
    for name in ("scipy", "statsmodels", "torch", "stable_baselines3", "gymnasium"):
        try:
            info[name] = importlib.import_module(name).__version__
        except ImportError:
            info[name] = None
    if info["torch"] is not None:
        import torch

        info["torch_num_threads"] = torch.get_num_threads()
        info["torch_build_config_sha256"] = hashlib.sha256(
            torch.__config__.show().encode()
        ).hexdigest()
    return info


def atomic_write_bytes(path, content: bytes):
    """Write next to the destination, fsync, then rename (no torn final files)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with open(temporary, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def json_bytes(value):
    """Strict JSON (no NaN/Infinity), identical to artifacts.write_json output."""
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def atomic_write_json(path, value):
    atomic_write_bytes(path, json_bytes(value))
