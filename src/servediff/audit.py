"""Which log lines servediff reads nothing from.

A tool that claims to show "what the server actually ran with" is defined as much by
the lines it ignores as by the ones it reads. This report is that gap: for each log,
every line no rule matched, deduplicated by its shape (digits replaced with N), with
counts. It is what an external user should paste into an issue when a fact comes out
`<absent>` and they suspect the line was there.

Most of that gap is noise — route tables, ASCII banners, deprecation warnings, progress
bars, access logs — which no fact could ever come from. Flooding the report with it is
how a report stops being read, so noise is counted and grouped separately from the
shapes that might actually be a missed configuration line.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from servediff.cli import ANSI, LINE_RULES, parse

DIGITS = re.compile(r"\d+")

# Lines whose content is never configuration. Each pattern must be narrow enough that a
# real config line cannot match it: widening one of these silently hides a missed fact.
# They are matched against the *raw* line, not the shape, because collapsing digits
# destroys the input they key on (a 200 status becomes N).
NOISE = (
    re.compile(r"Route: /\S*"),  # the HTTP route table
    re.compile(r"[\u2580-\u259f]{2,}"),  # box-drawing banners (██▄█▀)
    re.compile(r"\[\w+\.py:\d+\]\s*$"),  # banner lines that printed nothing after the tag
    re.compile(r"warnings\.warn\("),
    re.compile(r"DeprecationWarning|is deprecated, use"),
    re.compile(r"Loading (safetensors )?checkpoint shards"),  # tqdm progress bar
    re.compile(r'"(GET|POST) /[^"]*" \d{3}'),  # uvicorn access log
    re.compile(r"INFO:\s+(Started server process|Waiting for application startup)"),
    re.compile(r"Available routes are:"),
    # Python tracebacks: the root cause already lands in startup.error, the frames
    # themselves are never configuration.
    re.compile(r"^Traceback \(most recent call last\)"),
    re.compile(r'^\s*File "[^"]*", line \d+'),
    re.compile(r"^\s*[\^~]{3,}\s*$"),
    # The source lines a traceback echoes; each is part of a call, not a setting.
    re.compile(r"^\s*(self\.|return |raise )"),
    re.compile(r"^\s*[\w.]+ ?= ?[\w.]*\([^)]*\)\s*$"),
    re.compile(r"^\s*[\w.]+\([^)]*\)\s*$"),
    # Phase markers: they say a step began, never with what settings it ran.
    re.compile(r"JIT kernel warmup starting\.|Starting torch\.compile$"),
    # A per-machine absolute path, so it can never be equal across two boxes and no
    # setting hides inside it.
    re.compile(r"Using cache directory: |saved AOT compiled function to "),
    # Serving telemetry, printed while the server runs: a metric, not a startup setting.
    re.compile(r"Avg prompt throughput"),
)


def shape(line: str) -> str:
    """The line with numbers collapsed, so progress bars and ports group together."""
    return DIGITS.sub("N", ANSI.sub("", line).strip())


def unread(text: str) -> Counter:
    """Count the lines no rule reads, keyed by shape.

    Applies the same normalisation parse() does before matching, so a line appears here
    only when the parser really reads nothing from it.
    """
    return unread_samples(text)[0]


def unread_samples(text: str) -> tuple[Counter, dict[str, str]]:
    """(counts by shape, one raw line per shape), for classifying without losing digits."""
    counts: Counter = Counter()
    samples: dict[str, str] = {}
    for raw in text.splitlines():
        line = ANSI.sub("", raw)
        if not line.strip():
            continue
        if "non-default args:" in line or "server_args=" in line:
            continue  # consumed as argument dicts, not by line rules
        if any(pattern.search(line) for pattern, _ in LINE_RULES):
            continue
        key = shape(line)
        counts[key] += 1
        samples.setdefault(key, line.strip())
    return counts, samples


def is_noise(raw_line: str) -> bool:
    return any(pattern.search(raw_line) for pattern in NOISE)


def split(counts: Counter, samples: dict[str, str]) -> tuple[Counter, Counter]:
    """(shapes worth a look, noise), both preserving their counts."""
    noise: Counter = Counter()
    look: Counter = Counter()
    for line, count in counts.items():
        (noise if is_noise(samples.get(line, line)) else look)[line] = count
    return look, noise


def _rows(lines: list[str], counts: Counter) -> None:
    for line, count in counts.most_common()[:200]:
        lines.append(f"  {count:>4}  {line[:150]}")


def report(path: Path, text: str, limit: int = 0, everything: bool = False) -> str:
    """One block per log: totals first, then shapes, noise separated out."""
    counts, samples = unread_samples(text)
    lines = [f"{path}: {sum(counts.values())} unread line(s), {len(counts)} shape(s)"]
    if not counts:
        lines.append("  nothing unread: every non-empty line matched a rule")
        return "\n".join(lines)
    look, noise = split(counts, samples)
    if look:
        lines.append(f"  {sum(look.values())} line(s) worth a look — no rule reads them and"
                     " they are not routes, banners, deprecations, progress bars or access logs:")
        rows = look.most_common(limit or None)
        for line, count in rows:
            lines.append(f"  {count:>4}  {line[:150]}")
    else:
        lines.append("  nothing worth a look: every unread line is noise.")
    if noise and (everything or not look):
        lines.append(f"  {sum(noise.values())} line(s) of noise"
                     f" (routes, banners, deprecations, progress bars, access logs):")
        _rows(lines, noise)
    elif noise:
        lines.append(f"  {sum(noise.values())} line(s) of noise (routes, banners, deprecations,"
                     " progress bars, access logs); --everything to list them")
    return "\n".join(lines)
