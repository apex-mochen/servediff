"""Repeated log lines, counted by their cadence instead of by their number.

A server log repeats itself: an allocator retries an allocation in a burst, a ring-buffer
writer re-logs the same wait every warning interval. Counting the copies suggests N separate
failures and hides the only number that matters — the gap between them. `servediff repeats`
groups every message shape that appears more than once and reports its span, its median gap
and whether the gaps are constant.

A constant gap is a watchdog re-logging one continuous wait (vLLM's
`VLLM_RINGBUFFER_WARNING_INTERVAL` is 60 s, so sixteen copies of
"No available shared memory broadcast block found" are one sixteen-minute wait). An irregular
burst is a retry loop. The two need different fixes, and the line count cannot tell them apart.
"""

from __future__ import annotations

import statistics
import re
from collections import defaultdict
from pathlib import Path

from servediff.audit import message, shape
from servediff.cli import ANSI, BRACKET_TS

# The time a log line carries. vLLM writes `08-31 11:34:32`, a rank-prefixed torch warning
# writes `11:33:26.640394862`; only the clock is read, because the gap is what this report
# is about and a month-day only shifts it by days.
_CLOCK = re.compile(r"\d\d:\d\d:\d\d(?:[.,]\d+)?")

# 90 % of the gaps inside 5 % of the median: a metronome with one missed tick still counts
# as regular (vLLM's watchdog skips a tick when `elapsed > interval * n_warning` lands late),
# while a retry loop's 0.01 s / 1.0 s mixture never does.
_REGULAR_SHARE = 0.9
_REGULAR_TOLERANCE = 0.05
_REGULAR_FLOOR = 0.02  # seconds; comparing 60.0 with 60.0 should not have to be exact
# A watchdog re-logs at a human-scale period; an allocator retrying does it in milliseconds.
# Both can be perfectly even, so evenness alone does not make a group a watchdog — the
# magnitude does. Below this, an even cadence is a retry burst and gets the burst reading.
_REGULAR_MIN_GAP = 1.0

# A tqdm progress bar has no newline of its own: it rewrites the same terminal row with \r and
# a captured log keeps that, so the next process's line lands on the same physical line —
# `... 100%|█████| 51/51 [...] (EngineCore pid=260) INFO 08-31 11:37:30 shm_broadcast.py:801]`.
# Splitting there is what keeps that warning in its own shape instead of hiding it behind a
# progress bar's digits.
# The watchdog names its own period in the message: "found in 60 seconds". The timestamp on
# that line is the moment the wait *ended*, not the moment it began, which is why a run looks
# like it started waiting three seconds before its own graph capture finished.
_WAIT_PERIOD = re.compile(r"found in (\d+) seconds?")

_EMBEDDED = re.compile(r"(?=\([^)]*pid=\d+\)\s+(?:DEBUG|INFO|WARNING|ERROR|CRITICAL|VERBOSE)\b)")


def _clock(line: str) -> float | None:
    """Seconds of day carried by this line, or None when it has no clock."""
    match = _CLOCK.search(BRACKET_TS.sub("", ANSI.sub("", line)))
    if match is None:
        return None
    text = match.group(0).replace(",", ".")
    head, _, frac = text.partition(".")
    hour, minute, second = (int(part) for part in head.split(":"))
    return hour * 3600 + minute * 60 + second + (float("0." + frac) if frac else 0.0)


def _span(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.1f} min"


def _clock_text(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    whole = int(seconds)
    frac = seconds - whole
    text = f"{whole // 3600:02d}:{(whole % 3600) // 60:02d}:{whole % 60:02d}"
    return f"{text}.{int(round(frac * 1000)):03d}" if frac else text


def _timed(text: str) -> list[tuple[float, str, str]]:
    """(clock, shape, sample) for every message line that carries a clock.

    Used to answer the one question the cadence cannot: was anything else printed *during*
    the wait. A run compiling Triton kernels keeps writing inside the window; a run whose
    reader has stopped writes nothing at all.
    """
    out = []
    for raw in text.splitlines():
        for part in _EMBEDDED.split(raw):
            if not part.strip():
                continue
            body = message(part)
            if not body:
                continue
            clock = _clock(part)
            if clock is None:
                continue
            out.append((clock, shape(body), body))
    return out


def groups(text: str, min_count: int = 2) -> tuple[int, list[dict]]:
    """(lines read, repeated shapes with count >= min_count), ordered by count then shape."""
    buckets: dict[str, dict] = defaultdict(lambda: {"count": 0, "times": [], "samples": []})
    lines = 0
    for raw in text.splitlines():
        lines += 1
        for part in _EMBEDDED.split(raw):
            if not part.strip():
                continue
            body = message(part)
            if not body:
                continue
            key = shape(body)
            bucket = buckets[key]
            bucket["count"] += 1
            clock = _clock(part)
            if clock is not None:
                bucket["times"].append(clock)
            if len(bucket["samples"]) < 4 and body not in bucket["samples"]:
                bucket["samples"].append(body)

    found = []
    for key, bucket in buckets.items():
        if bucket["count"] < min_count:
            continue
        # Two ranks print the same startup line, and a captured log interleaves them:
        # `W831 11:33:27.693` can follow `11:33:27.834`. The gap is the question here, so
        # the clocks are read as a set of moments, not as a sequence of events.
        times = sorted(bucket["times"])
        gaps = [after - before for before, after in zip(times, times[1:])]
        median = statistics.median(gaps) if gaps else None
        regular = False
        if median is not None and len(gaps) >= 2 and median >= _REGULAR_MIN_GAP:
            floor = max(median * _REGULAR_TOLERANCE, _REGULAR_FLOOR)
            close = sum(1 for gap in gaps if abs(gap - median) <= floor)
            regular = close / len(gaps) >= _REGULAR_SHARE
        found.append({
            "message": key,
            "sample": bucket["samples"][0] if bucket["samples"] else key,
            "count": bucket["count"],
            "first": times[0] if times else None,
            "last": times[-1] if times else None,
            "span_s": (times[-1] - times[0]) if len(times) > 1 else 0.0,
            "median_gap_s": median,
            "max_gap_s": max(gaps) if gaps else None,
            "gaps": len(gaps),
            "regular": regular,
        })
    found.sort(key=lambda group: (-group["count"], group["message"]))
    return lines, found


def report(path: Path, text: str, min_count: int = 2, top: int = 0) -> tuple[int, list[dict], str]:
    """(lines read, every repeated shape, the block of text to print) for one log."""
    lines, found = groups(text, min_count)
    shown = found[:top] if top else found
    counted = sum(group["count"] for group in found)
    out = [f"{path}: {lines} line(s), {len(found)} shape(s) repeated"
           f" {min_count}+ time(s), {counted} line(s) in them"]
    if not found:
        out.append(f"  nothing repeats {min_count} or more times.")
    if top and len(found) > top:
        out.append(f"  showing the {top} most frequent; {len(found) - top} more below the cut.")
    timed = _timed(text) if any(group["regular"] for group in shown) else []
    for group in shown:
        if group["first"] is not None:
            where = (f"{_clock_text(group['first'])} -> {_clock_text(group['last'])}"
                     f"  span {_span(group['span_s'])}")
            cadence = "regular" if group["regular"] else "burst"
            gap = (f"{'gap' if group['regular'] else 'gaps'} "
                   f"{group['median_gap_s']:.3f} s median, {group['max_gap_s']:.3f} s max")
        else:
            where, cadence, gap = "no timestamps", "untimed", "gap unknown"
        out.append(f"  {group['count']:4d}x {cadence:8s} {where}")
        out.append(f"        {gap}")
        out.append(f"        {group['sample'][:150]}")
        if group["regular"] and group["first"] is not None:
            period = _WAIT_PERIOD.search(group["sample"])
            if period:
                seconds = int(period.group(1))
                began = (group["first"] - seconds) % 86400
                out.append(f"        this line is the end of a {seconds} s wait: the first one"
                           f" began at {_clock_text(began)}")
        if group["regular"] and group["last"] is not None:
            inside = [item for item in timed
                      if group["first"] < item[0] < group["last"] and item[1] != group["message"]]
            if inside:
                kinds = len({item[1] for item in inside})
                out.append(f"        {len(inside):4d} other line(s) inside that span,"
                           f" {kinds} shape(s): something was moving")
            else:
                out.append("          0 other line(s) inside that span:"
                           " nothing moved during the wait")
    if any(group["regular"] for group in found):
        out.append("")
        out.append("  A regular gap is one wait being re-logged, not N separate failures:"
                   " compare that span with anything that could have blocked a reader.")
    return lines, found, "\n".join(out)
