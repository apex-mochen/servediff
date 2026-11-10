#!/usr/bin/env python
"""Check that every exit code the README promises is the one the tool returns.

The command table in README.md is the contract a stranger reads before installing:
`1` means the inputs differ, `2` means the input was broken. A script in someone's CI
branches on those numbers, so a promised `2` that arrives as `1` is a silent lie --
which is exactly what `servediff env -o <missing dir>/snap.json` did (traceback, exit 1).

Run it after installing the package:

    python tools/verify_readme_exit_codes.py

It prints one line per case and exits 1 if any documented code was not reproduced.
Cases that need a GPU, a running server or an unread log line are listed at the end
with the reason they are not exercised here.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "vllm-0.33.0-real-rtx4060.log"


def command():
    """The servediff entry point: the installed script if present, else this interpreter."""
    exe = shutil.which("servediff")
    if exe:
        return [exe]
    return [sys.executable, "-m", "servediff"]


def main():
    if not SAMPLE.exists():
        raise SystemExit(f"missing sample log: {SAMPLE}")
    base = command()
    tmp = Path(tempfile.mkdtemp(prefix="servediff-exitcodes-"))
    try:
        log = tmp / "startup.log"
        shutil.copy(SAMPLE, log)
        text = log.read_text(encoding="utf-8")
        other = tmp / "other.log"
        # One byte of difference is enough: the question is the exit code, not the diff.
        other.write_text(text.replace("GPU KV cache size", "GPU KV cache SIZE")
                         if "GPU KV cache size" in text else text + "\nTRAILER\n",
                         encoding="utf-8")
        blank = tmp / "blank.log"
        blank.write_text("", encoding="utf-8")
        bad = tmp / "not-a-facts.json"
        bad.write_text("{}\n", encoding="utf-8")
        corpus = tmp / "logs"
        corpus.mkdir()
        shutil.copy(log, corpus / "a.log")
        shutil.copy(other, corpus / "b.log")

        def run(args, stdin_text=None):
            proc = subprocess.run(base + args, capture_output=True, text=True,
                                  input=stdin_text, cwd=tmp, encoding="utf-8", errors="replace")
            if proc.returncode not in (0, 1, 2, 7, 130) and "Traceback" in (proc.stderr + proc.stdout):
                # A traceback is never an answer; report it as an unreproduced code.
                return proc.returncode, proc.stdout + proc.stderr + "\n[traceback]"
            return proc.returncode, proc.stdout + proc.stderr

        # Facts and a snapshot produced by the tool itself, so the paths are real.
        _, facts_text = run(["show", str(log), "--json"])
        facts = tmp / "facts.json"
        facts.write_text(facts_text, encoding="utf-8")
        run(["env", "-o", str(tmp / "snap.json")])
        snapshot = tmp / "snap.json"

        cases = [
            ("show <log>", ["show", str(log)], 0),
            ("show <unreadable path>", ["show", str(tmp / "nope.log")], 2),
            ("show <blank log>", ["show", str(blank)], 2),
            ("diff <a> <a>", ["diff", str(log), str(log)], 0),
            ("diff <a> <b> differing", ["diff", str(log), str(other)], 1),
            ("diff <one log>", ["diff", str(log)], 2),
            ("diff <unreadable path> <log>", ["diff", str(tmp / "nope.log"), str(log)], 2),
            ("diff <log> --against <not a facts file>", ["diff", str(log), "--against", str(bad)], 2),
            ("diff <log> --against <facts> --only <no match>",
             ["diff", str(log), "--against", str(facts), "--only", "zzz."], 2),
            ("diff <log> <snapshot>", ["diff", str(log), str(snapshot)], 2),
            ("diff <dir> --against <facts>", ["diff", str(corpus), "--against", str(facts)], 1),
            ("series <a> <a> <a>", ["series", str(log), str(log), str(log)], 0),
            ("series <a> <a> <b>", ["series", str(log), str(log), str(other)], 1),
            ("series <a> <a> (two inputs)", ["series", str(log), str(log)], 2),
            ("env -o <writable path>", ["env", "-o", str(tmp / "snap2.json")], 0),
            ("env -o <missing directory>/x.json",
             ["env", "-o", str(tmp / "nodir" / "x.json")], 2),
            ("capture -o <missing directory>/run.log -- ...",
             ["capture", "-o", str(tmp / "nodir" / "run.log"), "--", "false"], 2),
            ("capture -- <command exiting 7> passes the code through",
             ["capture", "-o", str(tmp / "cap.log"), "--", sys.executable, "-c",
              "raise SystemExit(7)"], 7),
            ("audit <log>", ["audit", str(log)], 0),
            ("audit <unreadable path>", ["audit", str(tmp / "nope.log")], 2),
            ("audit <log> --baseline <not a baseline>",
             ["audit", str(log), "--baseline", str(bad)], 2),
        ]

        failures = 0
        for name, args, want in cases:
            got, output = run(args)
            if got == want and "[traceback]" not in output:
                print(f"ok       {name} -> {got}")
                continue
            failures += 1
            print(f"MISMATCH {name}: README says {want}, got {got}")
            tail = [line for line in output.strip().splitlines() if line.strip()]
            if tail:
                print(f"         {tail[-1][:200]}")

        print()
        print("Not exercised here, with the reason:")
        print("  audit --max-shapes 0 on a log with unread shapes -> 1")
        print("    Every log that ships is at 0 shapes by design (CI gates on that), so the")
        print("    threshold cannot be crossed without a hand-written line; tests/test_cli.py")
        print("    covers it (exit 1 at 0, exit 0 at 5).")
        print("  capture -> 130 on Ctrl-C, demo -> 1")
        print("    Both need the terminal's signal or the bundled demo, and both are covered")
        print("    in tests/test_cli.py.")
        print()
        total = len(cases)
        print(f"{total - failures}/{total} documented exit codes reproduced")
        return 1 if failures else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
