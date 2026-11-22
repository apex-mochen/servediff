import html

import pytest

pytest.importorskip("markdown", reason="build_site.py needs markdown; dev extra installs it")
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build():
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_site.py")],
                   cwd=ROOT, check=True, capture_output=True)
    return ROOT / "_site"


def test_every_writeup_gets_a_page_titled_with_its_first_line():
    """A note that renders nowhere is a note nobody reads. The page <title> is also the
    only organic search entry we control, so it has to be the headline, verbatim."""
    site = build()
    pages = sorted((ROOT / "writeups").glob("*.md"))
    assert pages, "no writeups to render"
    for md in pages:
        page = (site / (md.stem + ".html")).read_text(encoding="utf-8")
        title = re.search(r"<title>(.*?)</title>", page, re.S).group(1)
        headline = md.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        assert html.unescape(title) == re.sub(r"[`*_]", "", headline), (md.name, title, headline)
        assert len(page) > 2_000, (md.name, len(page))


def test_the_sitemap_lists_every_page():
    site = build()
    sitemap = (site / "sitemap.xml").read_text(encoding="utf-8")
    for md in (ROOT / "writeups").glob("*.md"):
        assert f"/servediff/{md.stem}.html" in sitemap, md.name
