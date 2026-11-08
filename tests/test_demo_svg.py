import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "render_demo_svg.py"


def test_svg_escapes_markup_and_keeps_rows_unwrapped(tmp_path):
    # A value like `<absent>` goes straight into the SVG; unescaped it breaks the document
    # and the README shows nothing at all. Table rows must not be wrapped either, or the
    # picture of a table becomes a picture of a paragraph.
    line = "kv_cache.tokens" + " " * 15 + "<absent> & more" + " " * 20 + "only_candidate"
    source = tmp_path / "out.txt"
    source.write_text("note: " + "long prose " * 20 + "\n" + line + "\n", encoding="utf-8")
    out = tmp_path / "demo.svg"
    subprocess.run([sys.executable, str(SCRIPT), str(source), str(out),
                    "--caption", "a & b"], check=True, capture_output=True)
    text = out.read_text(encoding="utf-8")
    ET.fromstring(text)  # parses only if escaping is right
    assert "&lt;absent&gt;" in text and "&amp; more" in text
    assert "<absent>" not in text
    rows = [l for l in text.splitlines() if "kv_cache.tokens" in l]
    assert len(rows) == 1, "a fact row must stay on one line"
    assert rows[0].rstrip().endswith("only_candidate</text>")


def test_committed_demo_svg_matches_the_demo_command(tmp_path):
    # The image in the README must be regenerable from the shipped demo, not hand-edited.
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    demo = subprocess.run([sys.executable, "-m", "servediff", "demo"],
                          cwd=ROOT, capture_output=True, text=True, env=env)
    assert demo.stdout.strip(), demo.stderr
    raw = tmp_path / "raw.txt"
    raw.write_text(demo.stdout, encoding="utf-8")
    svg = tmp_path / "fresh.svg"
    subprocess.run([sys.executable, str(SCRIPT), str(raw), str(svg)],
                   check=True, capture_output=True)
    committed = (ROOT / "assets" / "demo.svg").read_text(encoding="utf-8")
    for expected in ("backend.moe", "VLLM_CUTLASS", "MARLIN", "only_candidate",
                     "difference(s)"):
        assert expected in committed, expected
    assert expected in svg.read_text(encoding="utf-8")
