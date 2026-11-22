"""Print, or rewrite, the inventory of fact keys the bundled logs produce.

A rule that stops matching is silent: the run still succeeds, the table just has one row
fewer, and a diff between two logs can then report "no difference" for a setting that did
change. tests/test_cli.py compares against tests/fixtures/expected_fact_keys.txt, so a lost
rule fails the suite instead of shipping.
"""
import argparse
import sys
from pathlib import Path

from servediff import cli

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = FIXTURES / "expected_fact_keys.txt"


def keys() -> list[str]:
    found: set[str] = set()
    for log in sorted(FIXTURES.rglob("*.log")):
        found |= set(cli.parse(log.read_text(encoding="utf-8"))["facts"])
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite the golden file")
    args = ap.parse_args()
    ks = keys()
    if args.write:
        GOLDEN.write_text(
            "# Every fact key the bundled logs produce. Regenerate with:\n"
            "#   python tools/fact_keys.py --write\n" + "\n".join(ks) + "\n", encoding="utf-8")
        print(f"wrote {len(ks)} keys to {GOLDEN.relative_to(ROOT)}")
    else:
        print("\n".join(ks))
        print(f"# {len(ks)} keys", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
