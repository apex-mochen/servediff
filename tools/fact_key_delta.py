"""Compare the fact keys two installed versions read from the same logs.

A new rule is supposed to *add* keys, never to change what an older log reports. If it
accidentally matches a line it should not, the documented diffs move (53 differences
becomes 54) and nothing fails, because the exit code is still 1.

    python tools/fact_key_delta.py --before /tmp/v63/bin/servediff --after /tmp/v64/bin/servediff tests/fixtures

Exit code 1 when any log gains or loses a key, so it can be used as a release check.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def keys(exe: str, log: Path) -> set[str]:
    out = subprocess.run([exe, "show", "--json", str(log)], capture_output=True, text=True)
    # 0 = no facts, 1 = facts present, 2 = nothing recognised in this file; all three are
    # answers about the log, not failures of the tool.
    if out.returncode not in (0, 1, 2):
        raise SystemExit(f"{exe} failed on {log}: {out.stderr.strip()[:200]}")
    return set(json.loads(out.stdout or "{}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="servediff executable of the old version")
    ap.add_argument("--after", required=True, help="servediff executable of the new version")
    ap.add_argument("logs", nargs="+", help="log files or directories")
    args = ap.parse_args()

    files = []
    for item in args.logs:
        p = Path(item)
        files.extend(sorted(p.rglob("*.log")) if p.is_dir() else [p])

    changed = 0
    for log in files:
        a, b = keys(args.before, log), keys(args.after, log)
        if a == b:
            continue
        changed += 1
        print(f"{log.name}: +{len(b - a)} {sorted(b - a)} -{len(a - b)} {sorted(a - b)}")
    print(f"{len(files)} log(s) compared, {changed} changed")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
