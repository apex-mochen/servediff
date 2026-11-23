"""Build the wheel, install it into a fresh venv, and use it the way a stranger would.

Everything else in this repo tests the *source tree*. What ships is a wheel, and the
artifacts it needs at run time (`examples/` for `demo`) exist only if the packaging
config says so. This builds, installs with no index (so nothing can fall back to a
published copy), then checks:

  - `servediff --version` equals the version in pyproject.toml
  - bare `servediff` prints the first-run block and exits 0
  - `servediff demo` runs from a directory that has no source tree and exits 1
  - `show`/`audit`/`repeats`/`show --json` work on a real log copied out of tests/

    python tools/check_wheel.py --repo . --out /tmp/wheelcheck
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sys
import tempfile
import tomllib
from pathlib import Path


def run(cmd: list[str], cwd: Path, expect: int = 0) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=180)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != expect:
        raise SystemExit(f"FAIL: {' '.join(cmd)}\n  exit {proc.returncode}, wanted {expect}\n{out[-1200:]}")
    return proc.returncode, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default=None)
    parser.add_argument("--log", default="tests/fixtures/vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    meta = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = meta["version"]

    out = Path(args.out or tempfile.mkdtemp(prefix="wheelcheck-"))
    out.mkdir(parents=True, exist_ok=True)
    dist = out / "dist"
    shutil.rmtree(dist, ignore_errors=True)
    run([sys.executable, "-m", "build", "--wheel", "-o", str(dist), str(repo)], cwd=repo)
    wheel = next(dist.glob("*.whl"))
    print(f"built {wheel.name} ({wheel.stat().st_size} B), pyproject says {version}")

    venv = out / "venv"
    shutil.rmtree(venv, ignore_errors=True)
    run([sys.executable, "-m", "venv", str(venv)], cwd=out)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    run([str(py), "-m", "pip", "install", "-q", "--no-index", str(wheel)], cwd=out)

    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    shutil.copy(repo / args.log, work / "server.log")

    _, text = run([str(bin_dir / "servediff"), "--version"], cwd=work)
    if version not in text:
        raise SystemExit(f"FAIL: --version says {text.strip()!r}, pyproject says {version}")
    _, text = run([str(py), "-c", "import servediff; print(servediff.__version__)"], cwd=work)
    if version not in text:
        raise SystemExit(f"FAIL: __version__ says {text.strip()!r}, pyproject says {version}")

    code, text = run([str(bin_dir / "servediff")], cwd=work)
    if "servediff demo" not in text:
        raise SystemExit(f"FAIL: bare invocation printed no first-run block:\n{text}")

    _, text = run([str(bin_dir / "servediff"), "demo"], cwd=work, expect=1)
    if "MARLIN" not in text:
        raise SystemExit(f"FAIL: demo did not read its packaged logs:\n{text[-800:]}")

    run([str(bin_dir / "servediff"), "show", "server.log"], cwd=work)
    run([str(bin_dir / "servediff"), "audit", "server.log"], cwd=work)
    run([str(bin_dir / "servediff"), "repeats", "server.log"], cwd=work)
    _, text = run([str(bin_dir / "servediff"), "show", "server.log", "--json"], cwd=work)
    json.loads(text)

    # The glossary ships as a module, so it can be lost by a packaging change that drops
    # a subpackage; an installed `explain` is the proof it did not.
    _, text = run([str(bin_dir / "servediff"), "explain", "kv_cache.max_concurrency"], cwd=work)
    if "sequences" not in text:
        print("wheel: explain lost the glossary entry", file=sys.stderr)
        return 1
    _, text = run([str(bin_dir / "servediff"), "show", "server.log", "--evidence"], cwd=work)
    if "line 1:" not in text and "line " not in text:
        print("wheel: --evidence printed nothing", file=sys.stderr)
        return 1
    # The one-log report is what a stranger pastes into an issue; it has to survive
    # packaging too, and its table header is the cheapest proof that it rendered.
    _, text = run([str(bin_dir / "servediff"), "show", "server.log", "--markdown"], cwd=work)
    if "| Fact | Value | Line |" not in text:
        print("wheel: show --markdown did not render a table", file=sys.stderr)
        return 1

    print("wheel: version, first-run block, packaged demo, show/audit/repeats, --json, "
          "explain, --evidence, --markdown all pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
