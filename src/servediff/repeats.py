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

import bisect
import statistics
import re
from collections import defaultdict
from pathlib import Path

from servediff.audit import message, shape
from servediff.cli import ANSI, BRACKET_TS, _DIGITS, _normalize

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

# Two runs of the same server pasted into one file print the same warnings twice, minutes
# apart. The median gap between the copies still says "watchdog", but the span says
# "55 minutes", and nothing in the log lasted 55 minutes. A gap this large, and this much
# larger than the cadence, is a boundary between runs, not a tick.
_SEGMENT_MIN_GAP = 300.0
_SEGMENT_RATIO = 5.0

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
    # Search the raw line first: the clock digits survive normalisation untouched, so
    # running the preamble stripper over a 96 MB log only to read the same digits again is
    # wasted work on every line. Normalising stays as the fallback for a clock that an
    # escape sequence has split apart.
    match = _CLOCK.search(line)
    if match is None:
        match = _CLOCK.search(_normalize(line))
    if match is None:
        return None
    text = match.group(0).replace(",", ".")
    head, _, frac = text.partition(".")
    hour, minute, second = (int(part) for part in head.split(":"))
    return hour * 3600 + minute * 60 + second + (float("0." + frac) if frac else 0.0)


# The shape of a line depends only on the line minus its digits, and a server log repeats
# itself with new numbers: the same message a thousand times. Keeping the shape per
# digit-stripped line turns the whole normalisation (preamble removal, then digit collapse)
# into one dict hit, which is most of what a 96 MB `repeats` spent its time on.
_SHAPE_BY_STAMP: dict[str, str] = {}
_SHAPE_CACHE_LIMIT = 200_000


def _shape_of(part: str) -> tuple[str, str | None]:
    """(shape of the message, its body when this shape is new), keyed by the digits-free line.

    The body is returned only on a miss: a cache hit has already stored the sample text for
    that shape, so the caller does not need to re-run the preamble stripper.
    """
    stamp = _DIGITS.sub("0", part)
    if stamp in _SHAPE_BY_STAMP:
        return _SHAPE_BY_STAMP[stamp], None
    body = message(part)
    # An empty body is a line that carries no message at all; caching `None` records that
    # answer so the caller can skip it without running the stripper again.
    key = shape(body) if body else None
    if len(_SHAPE_BY_STAMP) < _SHAPE_CACHE_LIMIT:
        _SHAPE_BY_STAMP[stamp] = key
    return key, body


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


def groups(text: str, min_count: int = 2) -> tuple[int, list[dict]]:
    """(lines read, repeated shapes with count >= min_count), ordered by count then shape."""
    lines, _, found = _scan(text, min_count)
    return lines, found


def _scan(text: str, min_count: int = 2) -> tuple[int, dict, list[dict]]:
    """One pass over the log: (lines read, every shape's bucket, the repeated shapes).

    `report` needs the buckets as well as the repeats — the "was anything else printed during
    the wait" answer is a count of other shapes' timestamps inside a window, which is a bisect
    over timestamps already collected here. It used to be a second full pass over the whole
    log, which is why `repeats` was twice as expensive as `show` on the same file.
    """
    buckets: dict[str, dict] = defaultdict(lambda: {"count": 0, "times": [], "samples": []})
    lines = 0
    for raw in text.splitlines():
        lines += 1
        # A prompt line carrying several records is the only line that needs the lookahead
        # split; running it on every line was a regex per line for nothing.
        for part in (_EMBEDDED.split(raw) if "pid=" in raw else (raw,)):
            if not part.strip():
                continue
            key, body = _shape_of(part)
            if key is None:
                continue
            bucket = buckets[key]
            bucket["count"] += 1
            clock = _clock(part)
            if clock is not None:
                bucket["times"].append(clock)
            if len(bucket["samples"]) < 4:
                if body is None:
                    body = message(part)
                if body and body not in bucket["samples"]:
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
        # Where a run boundary falls: the copies on each side are still the same line, but
        # their spans are about different runs and must not be added together.
        cut = max(_SEGMENT_MIN_GAP, (median or 0.0) * _SEGMENT_RATIO)
        segments: list[list[float]] = []
        for clock in times:
            if segments and clock - segments[-1][-1] < cut:
                segments[-1].append(clock)
            else:
                segments.append([clock])
        longest = max((seg[-1] - seg[0] for seg in segments), default=0.0)
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
            "segments": len(segments),
            "longest_segment_s": longest,
        })
    found.sort(key=lambda group: (-group["count"], group["message"]))
    return lines, buckets, found


def _inside(buckets: dict, group: dict) -> tuple[int, int]:
    """(timed lines from other shapes strictly inside this group's window, how many shapes)."""
    first, last = group["first"], group["last"]
    count = 0
    kinds = 0
    for key, bucket in buckets.items():
        if key == group["message"]:
            continue
        times = bucket["times"]
        if not times:
            continue
        here = bisect.bisect_left(times, last) - bisect.bisect_right(times, first)
        if here:
            count += here
            kinds += 1
    return count, kinds


def report(path: Path, text: str, min_count: int = 2, top: int = 0) -> tuple[int, list[dict], str]:
    """(lines read, every repeated shape, the block of text to print) for one log."""
    lines, buckets, found = _scan(text, min_count)
    shown = found[:top] if top else found
    counted = sum(group["count"] for group in found)
    out = [f"{path}: {lines} line(s), {len(found)} shape(s) repeated"
           f" {min_count}+ time(s), {counted} line(s) in them"]
    if not found:
        out.append(f"  nothing repeats {min_count} or more times.")
    if top and len(found) > top:
        out.append(f"  showing the {top} most frequent; {len(found) - top} more below the cut.")
    for group in shown:
        if group["first"] is not None:
            where = (f"{_clock_text(group['first'])} -> {_clock_text(group['last'])}"
                     f"  span {_span(group['span_s'])}")
            if group.get("segments", 1) > 1:
                # The span crosses a gap between runs, so it is not how long anything lasted.
                where += (f" in {group['segments']} runs, longest {_span(group['longest_segment_s'])}")
                cadence = "split"
            else:
                cadence = "regular" if group["regular"] else "burst"
            if group["median_gap_s"] is None:
                # Count says 2+ but only one of the lines carried a clock, or the two
                # carried the same one. There is no gap to read; saying so is the answer.
                gap = f"{group['count']} line(s) but one clock: no gap to read"
            else:
                gap = (f"{'gap' if group['regular'] else 'gaps'} "
                       f"{group['median_gap_s']:.3f} s median, {group['max_gap_s']:.3f} s max")
        else:
            where, cadence, gap = "no timestamps", "untimed", "gap unknown"
        out.append(f"  {group['count']:4d}x {cadence:8s} {where}")
        out.append(f"        {gap}")
        out.append(f"        {group['sample'][:150]}")
        if group["regular"] and group["first"] is not None and group.get("segments", 1) <= 1:
            period = _WAIT_PERIOD.search(group["sample"])
            if period:
                seconds = int(period.group(1))
                began = (group["first"] - seconds) % 86400
                out.append(f"        this line is the end of a {seconds} s wait: the first one"
                           f" began at {_clock_text(began)}")
        if group["regular"] and group["last"] is not None and group.get("segments", 1) <= 1:
            inside, kinds = _inside(buckets, group)
            if inside:
                out.append(f"        {inside:4d} other line(s) inside that span,"
                           f" {kinds} shape(s): something was moving")
            else:
                out.append("          0 other line(s) inside that span:"
                           " nothing moved during the wait")
    if any(group["regular"] for group in found):
        out.append("")
        out.append("  A regular gap is one wait being re-logged, not N separate failures:"
                   " compare that span with anything that could have blocked a reader.")
    return lines, found, "\n".join(out)
