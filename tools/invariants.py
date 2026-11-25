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


def cli(repo, *args, stdin_text=None, stdin_bytes=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo, "src")
    data = stdin_bytes if stdin_bytes is not None else stdin_text
    as_text = not isinstance(data, (bytes, bytearray))
    # `-m servediff.cli` makes runpy warn that the module is already imported by the
    # package; `-m servediff` is the same entry point without the warning in every stderr.
    pd = subprocess.run([sys.executable, "-m", "servediff", *args],
                       capture_output=True, text=as_text, env=env, input=data)
    # Text mode turns `\r\n` into `\n` on the way in; the bytes path must be normalized the
    # same way or Windows would report every stdin comparison as a 62-character difference
    # (one `\r` per line) that is nothing but the harness reading its own output two ways.
    if as_text:
        return pd.returncode, pd.stdout.replace("\r\n", "\n"), pd.stderr.replace("\r\n", "\n")
    return (pd.returncode, pd.stdout.decode("utf-8", "replace").replace("\r\n", "\n"),
            pd.stderr.decode("utf-8", "replace").replace("\r\n", "\n"))


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


def check_stdin_matches_path(repo, logs):
    """`show file` and `cat file | show -` must print one table, not two.

    0.4.107: the gzip look at the first bytes lived in the file branch only, so a gzipped
    log on stdin was decoded as text and reported as "no configuration lines recognised".
    A transport is not a different reader, and neither is a container: both the plain bytes
    and the gzip of the same bytes are piped here, because the fix that covered one of them
    did not cover the other until 0.4.107.
    """
    import gzip
    bad = []
    for f in logs:
        raw = open(f, "rb").read()
        rc1, out1, _ = cli(repo, "show", f)
        for label, data in (("stdin", raw), ("stdin gz", gzip.compress(raw))):
            rc2, out2, err2 = cli(repo, "show", "-", stdin_bytes=data)
            if rc1 != rc2 or out1 != out2:
                bad.append((f, label, rc1, rc2, len(out1), len(out2), err2.strip()[:80]))
    return bad


def check_gz_matches_plain(repo, logs):
    """A .gz log and the same log uncompressed must read the same.

    0.4.107 was that they did not when the archive arrived through a pipe; this checks the
    on-disk half as well, so a future container change cannot fix one path and break the other.
    """
    import gzip
    bad = []
    for f in logs:
        rc_plain, out_plain, _ = cli(repo, "show", f)
        with tempfile.TemporaryDirectory() as d:
            packed = os.path.join(d, "serve.log.gz")
            with open(packed, "wb") as fh:
                fh.write(gzip.compress(open(f, "rb").read()))
            rc_gz, out_gz, err_gz = cli(repo, "show", packed)
        if rc_plain != rc_gz or out_plain != out_gz:
            bad.append((f, rc_plain, rc_gz, err_gz.strip()[:80]))
    return bad


def check_gz_facts_match_plain(repo, logs):
    """`--against facts.json` and `--against facts.json.gz` must give one answer.

    0.4.108: the facts-file reader decoded the archive and reported `not JSON: 'utf-8' codec
    can't decode byte 0x8b in position 1`, which is the gzip header described as bad JSON.
    """
    import gzip
    bad = []
    for f in logs:
        rc, out, _ = cli(repo, "show", f, "--json")
        if rc != 0:
            continue
        with tempfile.TemporaryDirectory() as d:
            facts = os.path.join(d, "facts.json")
            with open(facts, "w", encoding="utf-8") as fh:
                fh.write(out)
            packed = facts + ".gz"
            with open(packed, "wb") as fh:
                fh.write(gzip.compress(out.encode("utf-8")))
            rc_p, out_p, _ = cli(repo, "diff", f, "--against", facts)
            rc_g, out_g, err_g = cli(repo, "diff", f, "--against", packed)
        if rc_p != rc_g or out_p != out_g:
            bad.append((f, rc_p, rc_g, err_g.strip()[:80]))
    return bad


def check_rotations_are_logs(repo, logs):
    """A log named `serve.log.1` is the same log, not a file to drop.

    0.4.109: a directory only matched `*.log` and `*.log.gz`, so a corpus rotated by
    logrotate's default (`serve.log`, `serve.log.1`, `serve.log.2`, `serve.log.3.gz`) was
    audited one file out of four and the count was reported over the file it picked. Scope
    is an answer too: four copies of one log must read as four logs with one set of facts.
    """
    import gzip
    bad = []
    for f in logs:
        raw = open(f, "rb").read()
        rc_one, out_one, _ = cli(repo, "audit", f, "--json")
        if rc_one != 0:
            continue
        one = jload(out_one)["logs"][0]
        with tempfile.TemporaryDirectory() as d:
            for name in ("serve.log", "serve.log.1", "serve.log.2"):
                with open(os.path.join(d, name), "wb") as fh:
                    fh.write(raw)
            with open(os.path.join(d, "serve.log.3.gz"), "wb") as fh:
                fh.write(gzip.compress(raw))
            rc_dir, out_dir, err_dir = cli(repo, "audit", d, "--json")
        if rc_dir != 0:
            bad.append((f, "exit", rc_dir, err_dir.strip()[:80]))
            continue
        entries = jload(out_dir)["logs"]
        if len(entries) != 4:
            bad.append((f, "counted", len(entries), "serve.log .1 .2 .3.gz"))
            continue
        for entry in entries:
            if (entry["facts"], entry["shapes"]) != (one["facts"], one["shapes"]):
                bad.append((f, os.path.basename(entry["path"]),
                            entry["facts"], entry["shapes"], one["facts"], one["shapes"]))
    return bad


def run_cli(repo, *args, tries=3):
    """cli(), retried when the exit code is one this tool cannot produce.

    On Windows a child python is sometimes killed before it prints anything (a security
    scanner, not servediff): the run comes back with a code like 67503194 and empty stdout.
    That is a harness failure and must not be reported as a product one, so an implausible
    code is retried; a real answer (0, 1, 2) is never retried.
    """
    rc, out, err = cli(repo, *args)
    for _ in range(tries - 1):
        if rc in (0, 1, 2):
            return rc, out, err
        print("      note: cli exited %r with %d byte(s) of stdout; retrying" % (rc, len(out)),
              file=sys.stderr)
        rc, out, err = cli(repo, *args)
    return rc, out, err


def check_diff_gz_matches_plain(repo, logs):
    """Two archived logs must diff like the two logs they were made from.

    0.4.107 taught the readers to unwrap a container, and `show` was the path that got
    checked. diff and series read the same two files through the same reader, so the
    question is asked here as well: the archive is a transport, not a different log.
    """
    import gzip
    bad = []
    for a, b in zip(logs, logs[1:]):
        rc_p, out_p, _ = run_cli(repo, "diff", a, b)
        with tempfile.TemporaryDirectory() as d:
            pa = os.path.join(d, os.path.basename(a) + ".gz")
            pb = os.path.join(d, os.path.basename(b) + ".gz")
            for src, dst in ((a, pa), (b, pb)):
                with open(dst, "wb") as fh:
                    fh.write(gzip.compress(open(src, "rb").read()))
            rc_g, out_g, err_g = run_cli(repo, "diff", pa, pb)
        if rc_p != rc_g or out_p != out_g:
            bad.append((a, b, rc_p, rc_g, len(out_p), len(out_g), err_g.strip()[:80]))
    return bad


CHECKS = [
    ("self-diff is empty", check_self_diff, "log(s)"),
    ("diff is symmetric", check_symmetry, "log(s)"),
    ("text/json/markdown agree", check_channels, "log(s)"),
    ("repeats counts the file", check_lines, "log(s)"),
    ("diff json adds up", check_diff_json, "log(s)"),
    ("series is the union of diffs", check_series, "log(s)"),
    ("stdin and path agree", check_stdin_matches_path, "log(s)"),
    ("gz and plain agree", check_gz_matches_plain, "log(s)"),
    ("gz facts and plain facts agree", check_gz_facts_match_plain, "log(s)"),
    ("rotations are logs", check_rotations_are_logs, "log(s)"),
    ("diff on gz agrees with plain", check_diff_gz_matches_plain, "pair(s)"),
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
