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

# A real multi-process log prefixes every line with who printed it and from where:
# ``(Worker_TP8 pid=123) INFO 11-06 12:00:00 [cuda.py:118] ...``. The noise patterns below
# describe the *message*, so they are tested against the message too, not only against a
# bare line — otherwise a whole traceback reads as "worth a look" just because each frame
# carries a `(EngineCore pid=…) ERROR … [core.py:…]` prefix. Only a prefix is ever stripped;
# the message itself is passed through untouched.
_PREFIX = (
    re.compile(r"\([^)]*pid=\d+\)\s*"),
    re.compile(r"(?:DEBUG|INFO|WARNING|ERROR|CRITICAL|VERBOSE)\b[:\s-]*"),
    # vLLM prints `INFO 10-28 11:26:34` — month-day, no year.
    re.compile(r"(?:\d{4}-)?\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:[,.]\d{3})?\s*"),
    re.compile(r"^-+\s+"),
    re.compile(r"^\[?[\w./]+\.py:\d+\]?\s*-?\s*"),
)


def message(line: str) -> str:
    """The line with its process/time/module preamble removed."""
    text = ANSI.sub("", line).strip()
    for _ in range(12):
        before = text
        for pattern in _PREFIX:
            if pattern.match(text):
                text = pattern.sub("", text, count=1).strip()
        if text == before:
            break
    return text


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
    re.compile(r"^During handling of the above exception|^Process \w+:|^Call stack:"),
    # The source lines a traceback echoes; each is part of a call, not a setting.
    re.compile(r"^\s*(self\.|return |raise |yield |assert |del |super\(\)|with |async with"
               r"|await |next\(|import |from )"),
    re.compile(r"^\s*(?:[\w.]+(?:,\s*[\w.]+)*|[\w.]+:\s*[\w.\[\]]+)\s*=\s*(?:await\s+)?[\w.]+\s*\("),  # `x = Y(` frame echo
    re.compile(r"^\s*[\w.]+\([^()]*(?:\([^()]*\)[^()]*)*\)(?:\))?\s*$"),  # `uvloop.run(run_server(args))`
    re.compile(r"^\s*[\w.]+\($"),  # a call split across lines: `super().__init__(`
    # A logging failure's own dump: the traceback is about logging, not about the server.
    re.compile(r"^(--- )?Logging error ---"),
    re.compile(r"^Message: |^Arguments: "),
    # Engine teardown, printed on every shutdown, with the settings of nothing.
    re.compile(r"\[shutdown\]|\[shutdown\]|trigger received signal=SIGTERM"),
    # Phase markers: they say a step began, never with what settings it ran.
    re.compile(r"JIT kernel warmup starting\.|Starting torch\.compile$"),
    # A per-machine absolute path, so it can never be equal across two boxes and no
    # setting hides inside it.
    re.compile(r"Using cache directory: |saved AOT compiled function to "),
    # Serving telemetry, printed while the server runs: a metric, not a startup setting.
    re.compile(r"Avg prompt throughput"),
    # Inter-process plumbing warnings and resource teardown.
    re.compile(r"resource_tracker: There appear to be"),
    re.compile(r"destroy_process_group\(\) was not called"),
    re.compile(r"No available shared memory broadcast block found"),
    re.compile(r"Guessing device ID based on global rank"),
    re.compile(r"The client socket cannot be initialized to connect to"),
    # The traceback's own terminator, which carries no message of its own.
    re.compile(r"^(SystemExit|KeyboardInterrupt)$"),
    # uvicorn's own shutdown narration.
    re.compile(r"^(Shutting down|Waiting for application shutdown"
               r"|Application shutdown complete|Started server process)\.?$"),
    # Text pasted from a bug-report template rather than printed by the server.
    re.compile(r"^#{2,} |^- \[[ x]\] "),
    # Framework diagnostics that name their own source file before saying anything.
    re.compile(r"^\S+\.py:\d+: (User)?Warning:"),
    re.compile(r"\[Autotuner\]: Autotuning process starts"),
    # A wrapper script's own output, not the server's: the benchmark that launched vLLM
    # prints which config it is running. Auditing those lines finds no rule that could
    # ever read them, because they are not vLLM log lines.
    re.compile(r"^>>> |^\[spec_bench\]|^!!! "),
    # A bare `AssertionError` under a failed import check: the detail is the line above,
    # which is read as `kernel.module_missing` / `kernel.missing_lib`.
    re.compile(r"^AssertionError$"),
    re.compile(r"^I want to run inference of a \[specific model\]"),
)

# A line that is nothing but the prefix: `(EngineCore pid=3811)`.
_BARE_PREFIX = re.compile(r"^\s*\([^)]*pid=\d+\)\s*$")
_COMMENT = re.compile(r"\s+#\s.*$")


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
    if _BARE_PREFIX.match(raw_line):
        return True
    if any(pattern.search(raw_line) for pattern in NOISE):
        return True
    # The message without its preamble, and again with a trailing `# comment` gone: a
    # traceback echoes sources with comments in them (`_find_cuda_home()  # CUDA home`).
    stripped = message(raw_line)
    for candidate in (stripped, _COMMENT.sub("", stripped)):
        if candidate and any(pattern.search(candidate) for pattern in NOISE):
            return True
    return False


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
