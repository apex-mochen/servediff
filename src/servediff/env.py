"""Snapshot the environment a server runs in: package versions, GPU, relevant env vars.

Logs do not show dependency versions (torch, triton, flashinfer...), yet upgrades of
those change server behaviour. `snapshot()` runs a small stdlib-only probe under the
server's own interpreter, so nothing has to be installed into that environment.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile

SCHEMA = "servediff-env/1"

# Variables that steer inference engines. Values of names that look secret are redacted.
ENV_PREFIXES = ("VLLM_", "SGLANG_", "SGL_", "CUDA_", "NCCL_", "TORCH_", "PYTORCH_", "FLASHINFER_",
                "TRITON_", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
SECRET = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH", re.I)

# Runs under the target interpreter: standard library only.
PROBE = r"""
import importlib.metadata as m, json, os, platform, sys
prefixes = tuple(json.loads(sys.argv[1])); secret = __import__("re").compile(sys.argv[2], 2)
pkgs = {}
for d in m.distributions():
    name = (d.metadata["Name"] or "").lower().replace("_", "-")
    if name:
        pkgs[name] = d.version
env = {k: ("<redacted>" if secret.search(k) else v) for k, v in os.environ.items() if k.startswith(prefixes)}
print(json.dumps({"python": platform.python_version(), "executable": sys.executable,
                  "platform": platform.platform(), "packages": pkgs, "env": env}))
"""


def _gpus() -> list[dict] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        name, driver, memory = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "driver": driver, "memory_mib": int(float(memory))})
    return gpus


def snapshot(python: str | None = None) -> dict:
    # Not `-I`: it hides user site-packages, which the server itself would see.
    # A neutral cwd keeps the caller's directory out of sys.path.
    result = subprocess.run(
        [python or sys.executable, "-c", PROBE, json.dumps(ENV_PREFIXES), SECRET.pattern],
        capture_output=True, text=True, timeout=120, cwd=tempfile.gettempdir(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"environment probe failed under {python or sys.executable}: {result.stderr.strip()[-500:]}")
    data = json.loads(result.stdout)
    data["schema"] = SCHEMA
    data["gpus"] = _gpus()
    return data


def is_snapshot(text: str) -> bool:
    try:
        return json.loads(text).get("schema", "").startswith("servediff-env/")
    except (ValueError, AttributeError):
        return False


def diff(a: dict, b: dict) -> list[dict]:
    rows = []

    def add(key, va, vb):
        if va != vb:
            status = "only_candidate" if va is None else "only_baseline" if vb is None else "changed"
            rows.append({"key": key, "baseline": "<absent>" if va is None else va,
                         "candidate": "<absent>" if vb is None else vb, "status": status})

    add("python", a.get("python"), b.get("python"))
    ga = [(g["name"], g["driver"]) for g in a.get("gpus") or []]
    gb = [(g["name"], g["driver"]) for g in b.get("gpus") or []]
    add("gpus", ga or None, gb or None)
    for name in sorted(set(a["packages"]) | set(b["packages"])):
        add(f"pkg.{name}", a["packages"].get(name), b["packages"].get(name))
    for name in sorted(set(a["env"]) | set(b["env"])):
        add(f"env.{name}", a["env"].get(name), b["env"].get(name))
    return rows


# Packages whose version is most likely to change inference behaviour; shown first.
KEY_PACKAGES = re.compile(r"^pkg\.(vllm|sglang|sgl-kernel|torch|triton|flashinfer.*|xformers|deep-gemm|transformers|"
                          r"tokenizers|nvidia-.*|cuda-python|cuda-bindings|vllm-flash-attn|flash-attn|"
                          r"compressed-tensors|llguidance|xgrammar|outlines-core)$")


def is_key(row: dict) -> bool:
    key = row["key"]
    return key in ("python", "gpus") or key.startswith("env.") or bool(KEY_PACKAGES.match(key))


def ranked(rows: list[dict]) -> list[dict]:
    """Python and GPU first, then key inference packages and env vars, then everything else."""
    return sorted(rows, key=lambda r: (0 if r["key"] in ("python", "gpus") else 1 if is_key(r) else 2, r["key"]))
