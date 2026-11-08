"""Render a real `servediff` output into a terminal-styled SVG for the README.

The SVG is generated from captured stdout, never hand-drawn: if the demo output
changes, re-run this and the picture changes with it. Text is XML-escaped, so a
value containing `<absent>` or `&` cannot break the document.
"""

import argparse
import sys
from pathlib import Path

FONT_SIZE = 13
CHAR_W = FONT_SIZE * 0.6  # monospace advance, used only to size the window
LINE_H = 20
PAD_X = 22
PAD_Y = 18
BAR_H = 30

BG = "#0d1117"
BAR = "#161b22"
DOT = ("#f85149", "#d29922", "#3fb950")
FG = "#c9d1d9"
DIM = "#8b949e"
AMBER = "#e3b341"
BLUE = "#79c0ff"
GREEN = "#56d364"
BOLD = "#f0f6fc"


def escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def is_table_row(line: str) -> bool:
    # A fact row is padded into columns; wrapping one would look like a broken table.
    return bool(line) and line.rstrip().endswith(
        ("changed", "same", "only_candidate", "only_baseline", "status"))


def wrap(line: str, width: int) -> list[str]:
    """Hard-wrap prose to `width` columns. Purely a display choice: the SVG is a
    picture, and a 200-column terminal does not fit in a README."""
    if len(line) <= width or is_table_row(line):
        return [line]
    out, current, indent = [], "", ""
    for word in line.split(" "):
        candidate = f"{current} {word}" if current else f"{indent}{word}"
        if len(candidate) > width and current:
            out.append(current)
            indent = "  "
            current = f"{indent}{word}"
        else:
            current = candidate
    out.append(current)
    return out


def color_for(line: str) -> str:
    if line.startswith("note:"):
        return DIM
    if line.startswith("warning:"):
        return AMBER
    if line.startswith("servediff demo"):
        return BLUE
    if line.endswith("status") or "baseline" in line and "status" in line:
        return DIM
    if line.rstrip().endswith("changed"):
        return AMBER
    if line.rstrip().endswith("only_candidate") or line.rstrip().endswith("only_baseline"):
        return BLUE
    if line.rstrip().endswith("same"):
        return GREEN
    if line.startswith("2 difference") or "difference(s);" in line:
        return BOLD
    if line.startswith("demo:"):
        return DIM
    return FG


def render(lines: list[str], caption: str = "", width_cols: int = 96) -> str:
    # Colour is decided on the original line and inherited by its wrapped continuation,
    # otherwise the second half of a `note:` reads like body text.
    pairs = [(color_for(line), part)
             for line in lines
             for part in wrap(line, width_cols)]
    widest = max((len(t) for _, t in pairs), default=0)
    width = int(PAD_X * 2 + widest * CHAR_W)
    if caption:
        width = max(width, int(PAD_X * 2 + len(caption) * 11 * 0.62))
    width = max(width, 560)
    height = int(BAR_H + PAD_Y * 2 + len(pairs) * LINE_H + (18 if caption else 0))
    rows = []
    y = BAR_H + PAD_Y + LINE_H - 4
    for color, text in pairs:
        rows.append(
            f'<text x="{PAD_X}" y="{y}" fill="{color}" '
            f'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
            f'font-size="{FONT_SIZE}">{escape(text) or " "}</text>'
        )
        y += LINE_H
    if caption:
        rows.append(
            f'<text x="{PAD_X}" y="{y + 6}" fill="{DIM}" font-size="11" '
            f'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">'
            f"{escape(caption)}</text>"
        )
    dots = "".join(
        f'<circle cx="{PAD_X + i * 15}" cy="{BAR_H // 2}" r="5" fill="{c}"/>'
        for i, c in enumerate(DOT)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="{width}" height="{height}" rx="8" fill="{BG}"/>
  <rect width="{width}" height="{BAR_H}" rx="8" fill="{BAR}"/>
  <rect y="{BAR_H - 8}" width="{width}" height="8" fill="{BAR}"/>
{dots}
{chr(10).join(rows)}
</svg>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render servediff output as an SVG terminal window")
    ap.add_argument("source", type=Path, help="captured stdout of a servediff command")
    ap.add_argument("output", type=Path)
    ap.add_argument("--caption", default="")
    ap.add_argument("--wrap", type=int, default=96,
                    help="columns to wrap prose at (table rows are never wrapped)")
    args = ap.parse_args(argv)
    text = args.source.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
    lines = text.split("\n")
    args.output.write_text(render(lines, args.caption, args.wrap), encoding="utf-8")
    print(f"{args.output}: {len(lines)} line(s), {args.output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
