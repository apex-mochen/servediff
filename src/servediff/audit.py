"""Which log lines servediff reads nothing from.

A tool that claims to show "what the server actually ran with" is defined as much by
the lines it ignores as by the ones it reads. This report is that gap: for each log,
every line no rule matched, deduplicated by its shape (digits replaced with N), with
counts. It is what an external user should paste into an issue when a fact comes out
`<absent>` and they suspect the line was there.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from servediff.cli import ANSI, LINE_RULES, parse

DIGITS = re.compile(r"\d+")


def shape(line: str) -> str:
    """The line with numbers collapsed, so progress bars and ports group together."""
    return DIGITS.sub("N", ANSI.sub("", line).strip())


def unread(text: str) -> Counter:
    """Count the lines no rule reads, keyed by shape.

    Applies the same normalisation parse() does before matching, so a line appears here
    only when the parser really reads nothing from it.
    """
    counts: Counter = Counter()
    for raw in text.splitlines():
        line = ANSI.sub("", raw)
        if not line.strip():
            continue
        if "non-default args:" in line or "server_args=" in line:
            continue  # consumed as argument dicts, not by line rules
        if any(pattern.search(line) for pattern, _ in LINE_RULES):
            continue
        counts[shape(line)] += 1
    return counts


def report(path: Path, text: str, limit: int = 0) -> str:
    """One block per log: totals first, then the most common unread shapes."""
    counts = unread(text)
    lines = [f"{path}: {sum(counts.values())} unread line(s), {len(counts)} shape(s)"]
    if not counts:
        lines.append("  nothing unread: every non-empty line matched a rule")
        return "\n".join(lines)
    for line, count in counts.most_common(limit or None):
        lines.append(f"  {count:>4}  {line[:150]}")
    return "\n".join(lines)
