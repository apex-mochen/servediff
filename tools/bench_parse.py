"""Generate a big synthetic log and time `show`, `audit` and `repeats` on it.

The point is to keep the "how long does this take on a log a stranger actually
has" number reproducible instead of remembered. The synthetic lines are shaped
like real vLLM lines (timestamped preamble + `[module.py:NN]` marker) and cycle
through five distinct shapes, so both the rule loop and the shape cache are
exercised.

    python tools/bench_parse.py --lines 1456284 --out /tmp/big.log
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

SHAPES = [
    "[{mod}:{line}] INFO {stamp} {what} {value:.2f} GiB",
    "[{mod}:{line}] INFO {stamp} Capturing CUDA graphs ({mode}): {pct:3d}%|          | {done}/{total} [{m:02d}:{s:02d}<?, ?it/s]",
    "[{mod}:{line}] INFO {stamp} Engine {engine} started with {value} blocks",
    "[{mod}:{line}] WARNING {stamp} It took {ms:.3f} seconds to {what}",
    "[{mod}:{line}] INFO {stamp} {what} took {value:.2f} s, {count} items processed",
]


def line(i: int) -> str:
    shape = SHAPES[i % len(SHAPES)]
    hour, minute, second = 22, (i // 4000) % 60, i % 60
    return shape.format(
        mod=f"mod{i % 37}",
        line=100 + i % 800,
        stamp=f"[{hour:02d}:{minute:02d}:{second:02d}]",
        what=f"step_{i % 11}",
        value=1.5 + (i % 97) / 10,
        mode="FULL" if i % 2 else "PIECEWISE",
        pct=(i % 101),
        done=i % 3,
        total=3,
        m=minute,
        s=second,
        engine=i % 5,
        ms=0.25 + (i % 13),
        count=i % 7,
    )


def build(path: str, count: int) -> int:
    written = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for i in range(count):
            text = line(i) + "\n"
            handle.write(text)
            written += len(text.encode("utf-8"))
    return written


def run(command: str, path: str) -> float:
    env = dict(os.environ)
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "servediff.cli", command, path],
        capture_output=True,
        env=env,
    )
    return time.time() - started, proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=int, default=1_456_284)
    parser.add_argument("--out", default="/tmp/bench.log")
    args = parser.parse_args()

    size = build(args.out, args.lines)
    print(f"{args.lines} lines, {size / 1e6:.1f} MB -> {args.out}")
    for command in ("show", "audit", "repeats"):
        seconds, code = run(command, args.out)
        print(f"{command:8s} {seconds:6.2f}s  exit={code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
