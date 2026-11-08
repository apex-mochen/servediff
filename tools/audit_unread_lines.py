"""Dev entry point for the unread-line audit; the shipped command is `servediff audit`.

Kept so `python tools/audit_unread_lines.py tests/fixtures/*.log` still works from a
checkout, and so adding a rule can be checked against every fixture in one run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from servediff.audit import report  # noqa: E402


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv[1:]]
    if not paths:
        paths = sorted(Path("tests/fixtures").glob("*.log"))
    blocks = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks.append(report(path, text))
    print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
