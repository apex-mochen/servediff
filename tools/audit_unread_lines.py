"""List log lines that servediff currently reads nothing from.

A tool that claims to show "what the server actually ran with" is defined as much by
the lines it ignores as by the ones it reads. This report is the gap: for every fixture,
the lines no rule matched, deduplicated by their shape, with the counts. It is a
diagnostic, not a test -- run it after adding a rule to see the gap shrink.

    python tools/audit_unread_lines.py tests/fixtures/*.log
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from servediff.cli import ANSI, LINE_RULES, parse  # noqa: E402

TAIL = re.compile(r"\d+")


def shape(line: str) -> str:
    return TAIL.sub("N", ANSI.sub("", line).strip())


def audit(path: Path) -> Counter:
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = parse(text)
    engine = parsed["engine"]
    unread: Counter = Counter()
    for raw in text.splitlines():
        # Same normalisation parse() applies, so a line is "unread" here only when the
        # parser really reads nothing from it.
        line = ANSI.sub("", raw)
        if not line.strip():
            continue
        if "non-default args:" in line or "server_args=" in line:
            continue
        if any(pattern.search(line) for pattern, _ in LINE_RULES):
            continue
        unread[shape(line)] += 1
    return unread


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv[1:]]
    if not paths:
        paths = sorted(Path("tests/fixtures").glob("*.log"))
    for path in paths:
        unread = audit(path)
        total = sum(unread.values())
        print(f"\n=== {path.name}: {total} unread line(s), {len(unread)} shape(s)")
        for line, count in unread.most_common():
            print(f"{count:>4}  {line[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
