"""Render writeups/ into a static site for GitHub Pages.

The notes are written in the repo and linked from the README, but the repo itself is not in
GitHub code search and pypi.org does not host them. A page whose <title> is the exact error
string people paste into a search box is the only organic entry point we control.
"""
import html
import re
import sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from servediff.glossary import GLOSSARY, families  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_site"

CSS = """
:root { --fg:#1c1c1e; --muted:#6b6b70; --line:#e4e4e7; --code:#f6f6f7; }
body { margin:0 auto; max-width:44rem; padding:2.5rem 1.25rem 6rem;
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  color:var(--fg); }
h1 { font-size:1.9rem; line-height:1.25; margin:0 0 1rem; letter-spacing:-0.01em; }
h2 { font-size:1.25rem; margin:2.2rem 0 .6rem; }
h3 { font-size:1.05rem; margin:1.6rem 0 .4rem; }
p, li { color:var(--fg); }
pre { background:var(--code); border:1px solid var(--line); border-radius:6px;
  padding:.85rem 1rem; overflow-x:auto; font-size:.86rem; line-height:1.5; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.9em; }
p code, li code { background:var(--code); padding:.1em .3em; border-radius:3px; }
a { color:#0b62d0; }
.note { color:var(--muted); font-size:.9rem; }
nav { border-bottom:1px solid var(--line); padding-bottom:.75rem; margin-bottom:2rem;
  font-size:.9rem; }
nav a { margin-right:1rem; }
ul.index li { margin:.5rem 0; }
"""


def page(title: str, body: str, description: str, canonical: str, root: str = ".") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="canonical" href="{canonical}">
<style>{CSS}</style>
</head>
<body>
<nav><a href="{root}/index.html">servediff notes</a><a href="https://github.com/apex-mochen/servediff">source</a><a href="https://pypi.org/project/servediff/">pip install servediff</a></nav>
{body}
</body>
</html>
"""


def first_paragraph(text: str) -> str:
    for block in text.split("\n\n"):
        b = block.strip()
        if b and not b.startswith(("#", "```", "|", "-", "*", "1.")):
            return re.sub(r"[`*_]", "", " ".join(b.split()))[:220]
    return ""


def main() -> int:
    OUT.mkdir(exist_ok=True)
    notes = []
    for md in sorted((ROOT / "writeups").glob("*.md")):
        text = md.read_text(encoding="utf-8")
        raw_title = text.splitlines()[0].lstrip("# ").strip()
        title = re.sub(r"[`*]", "", raw_title)
        body = markdown.markdown(text, extensions=["fenced_code", "tables"])
        slug = md.stem + ".html"
        desc = first_paragraph("\n".join(text.splitlines()[1:]))
        (OUT / slug).write_text(
            page(title, body, desc, f"https://apex-mochen.github.io/servediff/{slug}"),
            encoding="utf-8")
        notes.append((title, slug, desc))
        print(f"{slug}: {len(body)} chars, title={title[:60]!r}")

    # One page for the vocabulary itself: `show` prints keys, and the meaning of each used
    # to live in a source comment. Search traffic lands on the error string, but the
    # second thing a reader needs is "what does this number count".
    fam = families()
    exact = sorted(k for k in GLOSSARY if "." in k)
    parts = ["<h1>What each servediff fact key means</h1>",
             "<p>Each entry is one key, or one family when every key in it shares a "
             "meaning. The third paragraph is the one that matters: what the number "
             "does <em>not</em> answer, which is how a diff gets misread.</p>",
             "<pre><code>servediff explain kv_cache.max_concurrency</code></pre>",
             "<ul class=\"index\">"]
    for name, g in fam:
        parts.append(f'<li><a href="#{name}"><code>{name}</code></a> '
                     f'&mdash; {html.escape(g.what.split(". ")[0])}</li>')
    parts.append("</ul>")
    for name, g in fam:
        members = [k for k in exact if k.split(".")[0] == name]
        parts.append(f'<h2 id="{name}"><code>{name}</code></h2>')
        parts.append(f"<p><strong>Measured in:</strong> {html.escape(g.unit)}</p>")
        parts.append(f"<p>{html.escape(g.what)}</p>")
        if g.trap:
            parts.append(f'<p class="note">What it does not tell you: {html.escape(g.trap)}</p>')
        for k in members:
            kg = GLOSSARY[k]
            parts.append(f'<h3><code>{k}</code></h3>')
            parts.append(f"<p><strong>Measured in:</strong> {html.escape(kg.unit)}</p>")
            parts.append(f"<p>{html.escape(kg.what)}</p>")
            if kg.trap:
                parts.append(f'<p class="note">What it does not tell you: {html.escape(kg.trap)}</p>')
    vocab = "\n".join(parts)
    (OUT / "facts.html").write_text(
        page("What each servediff fact key means",
             vocab,
             "Every key servediff prints from a vLLM or SGLang startup log, with what it "
             "counts and what it does not tell you.",
             "https://apex-mochen.github.io/servediff/facts.html"),
        encoding="utf-8")
    print(f"facts.html: {len(vocab)} chars, {len(fam)} families, {len(exact)} exact keys")
    notes.append(("What each fact key means",
                  "facts.html",
                  "Every key servediff prints, with what it counts and what it does not tell you."))

    items = "\n".join(
        f'<li><a href="{s}">{html.escape(t)}</a><br><span class="note">{html.escape(d)}</span></li>'
        for t, s, d in notes)
    index = f"""<h1>servediff notes: startup logs, read as facts</h1>
<p><code>servediff</code> reads a vLLM or SGLang <em>startup log</em> into named facts
(&ldquo;<code>backend.moe</code> was <code>VLLM_CUTLASS</code> in one run and
<code>MARLIN</code> in the other&rdquo;) so an upgrade can be compared on what the server
actually ran with, not on the flags you passed.</p>
<pre><code>pip install servediff
servediff demo</code></pre>
<p>These notes are the long-form evidence behind the rules: real logs, the arithmetic, and
what the popular answer gets wrong.</p>
<ul class="index">
{items}
</ul>
<p class="note">Source: <a href="https://github.com/apex-mochen/servediff">github.com/apex-mochen/servediff</a>.
Pages are generated from <code>writeups/*.md</code> by <code>tools/build_site.py</code>.</p>"""
    (OUT / "index.html").write_text(
        page("servediff notes: vLLM / SGLang startup logs, read as facts",
             index,
             "Read a vLLM or SGLang startup log as named facts and diff two runs: "
             "auto-selected backends, KV cache layout, CUDA graph capture, startup failures.",
             "https://apex-mochen.github.io/servediff/"),
        encoding="utf-8")
    (OUT / "robots.txt").write_text("User-agent: *\nAllow: /\nSitemap: "
                                    "https://apex-mochen.github.io/servediff/sitemap.xml\n",
                                    encoding="utf-8")
    entries = "".join(f"<url><loc>https://apex-mochen.github.io/servediff/{s}</loc></url>\n"
                      for _, s, _ in notes)
    (OUT / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        "<url><loc>https://apex-mochen.github.io/servediff/</loc></url>\n"
        f"{entries}</urlset>\n", encoding="utf-8")
    print("index.html, robots.txt, sitemap.xml written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
