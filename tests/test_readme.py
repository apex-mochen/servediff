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


def test_readme_names_the_facts_it_ships():
    """A fact in the matrix is a promise; a fact missing from it is one users cannot find.

    The 0.4.64 CPU facts were read by rules before they had a row here, which is the same
    kind of gap as the dead PyPI links: true in the code, absent where people look.
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for fact in (
        "kv_cache.mamba_page_pad",
        "kv_cache.cpu_node",
        "kv_cache.lcm_block_sizes",
        "mm.encoder_cache",
        "compile.inductor_disabled",
        "api.chat_template_order",
        "cpu.numa_membind",
    ):
        assert fact in text, fact


def test_the_chinese_readme_points_at_the_english_one_and_the_notes_site():
    """The Chinese page is an introduction, not a translation, so it drifts silently.

    Two things must not drift: where the full list lives, and where the rendered writeups
    are. Both were missing while the English page had gained a section on the notes site.
    """
    text = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert "blob/main/README.md" in text
    assert "apex-mochen.github.io/servediff" in text
    assert "sglang-2026-10" in text
