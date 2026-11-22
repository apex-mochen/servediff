import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_readme_links_to_repo_files_are_absolute():
    """PyPI renders the README against pypi.org, so a relative link points at nothing.

    The project page is the first thing a pip user sees; three writeups links sat there
    dead for two releases before anyone clicked one.
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    relative = [t for t in re.findall(r"\]\(([^)]+)\)", text)
                if not t.startswith(("http://", "https://", "#", "mailto:"))]
    assert relative == []


def test_readme_mentions_every_writeup():
    """A writeup that is not in the index is a file nobody opens."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    writeups = sorted(p.name for p in (ROOT / "writeups").glob("*.md"))
    missing = [w for w in writeups if w not in text]
    assert missing == []
