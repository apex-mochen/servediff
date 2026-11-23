"""Cross-command invariants over the log corpus: one command, one table.

Each command answers a question about the same log from a different angle, and they must not
disagree. Everything below was a real defect at some point (0.4.97 counted a line as compared
when it was only printed on one side; 0.4.98 let `after_ready` rows change a `series` count;
0.4.99 attributed a later run's waits to this startup; 0.4.100 measured the gap between two
runs as a span). The checks are written to fail on that shape of mistake, not on a version.

Usage:
    python tools/invariants.py --repo .                 # whole corpus, ~1 min
    python tools/invariants.py --repo . --sample 20     # a slice, for a quick loop
Exit code 0 when every check passes, 1 on the first check with a violation (the offending
log is printed).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

OK = "ok"


def cli(repo, *args, stdin_text=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo, "src")
    p = subprocess.run([sys.executable, "-m", "servediff.cli", *args],
                       capture_output=True, text=True, env=env, input=stdin_text)
    return p.returncode, p.stdout, p.stderr


def jload(out):
    """Parse the command's JSON, or None: a crash is a violation, not a traceback in this tool."""
    try:
        return json.loads(out)
    except ValueError:
        return None


def logs_of(repo):
    return sorted(glob.glob(os.path.join(repo, "tests", "fixtures", "**", "*.log"), recursive=True))


def sel(logs, n):
    if not n or n >= len(logs):
        return logs
    step = max(1, len(logs) // n)
    return logs[::step][:n]


def check_self_diff(repo, logs):
    """A log against itself has nothing to report, with or without --skip-absent."""
    bad = []
    for f in logs:
        for extra in ([], ["--skip-absent"]):
            rc, out, _ = cli(repo, "diff", f, f, "--json", *extra)
            if rc == 2:
                continue  # nothing recognised: not this check's business
            d = jload(out)
            if d is None:
                bad.append((f, extra, "no json"))
            elif d["differences"] != 0:
                bad.append((f, extra, d["differences"]))
    return bad


def check_symmetry(repo, logs):
    """Swapping the two logs swaps the sides, never the counts."""
    bad = []
    for a, b in zip(logs, logs[1:]):
        d1, d2 = jload(cli(repo, "diff", a, b, "--json")[1]), jload(cli(repo, "diff", b, a, "--json")[1])
        if d1 is None or d2 is None:
            bad.append((a, b, "no json"))
        elif (d1["differences"], d1["compared_both_sides"]) != (d2["differences"], d2["compared_both_sides"]):
            bad.append((a, b, d1["differences"], d2["differences"]))
    return bad


def check_channels(repo, logs):
    """text, --json and --markdown are three renderings of one fact set."""
    bad = []
    for f in logs:
        rc, txt, _ = cli(repo, "show", f)
        if rc == 2:
            continue
        _, js, _ = cli(repo, "show", f, "--json")
        _, md, _ = cli(repo, "show", f, "--markdown")
        n_txt = len([l for l in txt.splitlines() if l.strip() and not l.startswith("note")])
        d = jload(js)
        n_js = len(d) if d is not None else -1
        n_md = len([l for l in md.splitlines() if l.strip().startswith("|")]) - 2
        if not (n_txt == n_js == n_md):
            bad.append((f, n_txt, n_js, n_md))
    return bad


def check_lines(repo, logs):
    """`repeats` counts the file, not the lines it happened to parse."""
    bad = []
    for f in logs:
        _, out, _ = cli(repo, "repeats", f, "--json", "--top", "0")
        d = jload(out)
        if d is None:
            bad.append((f, "no json", 0))
            continue
        d = d[0]
        real = len(open(f, "rb").read().splitlines())
        if d["lines"] != real:
            bad.append((f, d["lines"], real))
    return bad


def check_diff_json(repo, logs):
    """differences == compared_both_sides + hidden_absent, whatever --skip-absent says."""
    bad = []
    for a, b in zip(logs, logs[1:]):
        for extra in ([], ["--skip-absent"]):
            rc, out, _ = cli(repo, "diff", a, b, "--json", *extra)
            if rc == 2:
                continue
            d = jload(out)
            if d is None:
                bad.append((a, b, extra, "no json"))
            elif d["differences"] != d["compared_both_sides"] + d["hidden_absent"]:
                bad.append((a, b, extra, d["differences"], d["compared_both_sides"], d["hidden_absent"]))
    return bad


def check_series(repo, logs):
    """Three logs: the series keys are the union of the pairwise diff keys."""
    bad = []
    triples = [logs[i:i + 3] for i in range(0, max(0, len(logs) - 2), 8)]
    for t in triples:
        if len(t) < 3:
            continue
        _, out, _ = cli(repo, "series", *t, "--json")
        s = jload(out)
        if s is None:
            bad.append((t, "no json", 0))
            continue
        keys = set()
        for a, b in ((t[0], t[1]), (t[0], t[2]), (t[1], t[2])):
            rc, dout, _ = cli(repo, "diff", a, b, "--json")
            if rc == 2:
                keys = None
                break
            dd = jload(dout)
            if dd is None:
                keys = None
                break
            keys |= {r["key"] for r in dd["rows"] if r["status"] != "same"}
        if keys is None:
            continue
        series_keys = {r["key"] for r in s["rows"] if r["status"] != "same"}
        if series_keys != keys:
            bad.append((t, len(series_keys), len(keys)))
    return bad


CHECKS = [
    ("self-diff is empty", check_self_diff, "log(s)"),
    ("diff is symmetric", check_symmetry, "log(s)"),
    ("text/json/markdown agree", check_channels, "log(s)"),
    ("repeats counts the file", check_lines, "log(s)"),
    ("diff json adds up", check_diff_json, "log(s)"),
    ("series is the union of diffs", check_series, "log(s)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--sample", type=int, default=0, help="use at most N logs (0 = all)")
    ap.add_argument("--only", action="append", default=None, help="run one check by name prefix")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    logs = sel(logs_of(repo), a.sample)
    print("%d log(s) under %s" % (len(logs), os.path.join(repo, "tests", "fixtures")))
    failed = 0
    for name, fn, unit in CHECKS:
        if a.only and not any(name.startswith(p) for p in a.only):
            continue
        bad = fn(repo, logs)
        if bad:
            failed += 1
            print("FAIL  %-30s %d %s" % (name, len(bad), unit))
            for item in bad[:3]:
                print("        %s" % (item,))
        else:
            print("ok    %-30s %d %s" % (name, len(logs), unit))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
