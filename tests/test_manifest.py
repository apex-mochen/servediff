from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_fixture_file_type_is_shipped_in_the_sdist():
    """The publish workflow runs the suite from the unpacked sdist, so a file a test opens
    but MANIFEST.in does not name makes the release fail at the last step.

    `expected_fact_keys.txt` was exactly that: one test opened it, `.txt` was not in the
    manifest, and the sdist suite failed while the repo suite passed. Checking the
    extensions under the fixtures tree catches the next one before it ships.
    """
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    patterns: dict[str, set[str]] = {}
    for line in manifest:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "recursive-include":
            patterns.setdefault(parts[1], set()).update(parts[2:])

    for directory in ("tests/fixtures", "examples"):
        shipped = patterns.get(directory, set())
        assert shipped, f"MANIFEST.in ships nothing from {directory}"
        suffixes = sorted({p.suffix for p in (ROOT / directory).rglob("*") if p.is_file()})
        missing = [s for s in suffixes if s and f"*{s}" not in shipped]
        assert missing == [], f"{directory}: {missing} not in MANIFEST.in"
