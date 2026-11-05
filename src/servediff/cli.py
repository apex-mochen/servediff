"""servediff: what did the inference server actually run with? Diff two server logs.

Benchmark result files record the client side only. The server's effective
configuration -- auto-selected attention/MoE backends, CUDA graph and compile
mode, the arguments it resolved -- is only printed in its startup log. This tool
extracts those facts from vLLM and SGLang logs and diffs two of them.

Every fact keeps the line it came from. A fact that is not in the log is reported
as absent, never filled with a default.

Usage:
    servediff show <log>
    servediff diff <baseline.log> <candidate.log> [--raw-args] [--json] [--markdown] [--all]
    servediff env [--python /path/to/server/venv/bin/python] [-o env.json]
    servediff diff <baseline-env.json> <candidate-env.json>
    servediff capture [-o run.log] [--python /path/to/python] -- <server command>
Exit codes: 0 no differences, 1 differences found, 2 nothing recognised in an input.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from servediff import env as envsnap

ANSI = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class Fact:
    values: list = field(default_factory=list)  # distinct values in order seen
    evidence: list = field(default_factory=list)  # (line number, line text)

    def add(self, value, lineno: int, line: str) -> None:
        if value not in self.values:
            self.values.append(value)
            self.evidence.append((lineno, line.strip()[:300]))

    @property
    def value(self):
        return self.values[0] if len(self.values) == 1 else self.values


# Patterns over one log line. Each yields (fact key, value) pairs.
LINE_RULES: list[tuple[re.Pattern, callable]] = [
    # vLLM banner and engine init.
    (re.compile(r"\bversion (\d+\.\d+\.\d+\S*)"), lambda m: [("engine.version", m[1])]),
    (re.compile(r"Initializing a V1 LLM engine \(v([^)]+)\)"), lambda m: [("engine.version", m[1])]),
    # vLLM: "Using FLASH_ATTN attention backend out of potential backends: [...]".
    (re.compile(r"Using (\S+) attention backend"), lambda m: [("backend.attention", m[1])]),
    (re.compile(r"Using FlashAttention version (\d+)"), lambda m: [("backend.flash_attn_version", m[1])]),
    # vLLM MoE selection, three formats seen in the wild:
    #   Using 'VLLM_CUTLASS' NvFp4 MoE backend
    #   Using FlashInfer CUTLASS Unquantized MoE backend out of potential backends: [...]
    #   Using TrtLlmBf16ExpertsMonolithic MoE backend
    (re.compile(r"Using '?([^']+?)'? (?:(NvFp4|Fp8|Unquantized|Mxfp4|Int4|W4A16) )?MoE backend"),
     lambda m: [("backend.moe", m[1] if m[2] is None else f"{m[1]} ({m[2]})")]),
    (re.compile(r"Using (MoEPrepareAndFinalize\w+)"), lambda m: [("backend.moe_prepare_finalize", m[1])]),
    (re.compile(r"Enforce eager set"), lambda m: [("cuda_graph", "disabled (enforce_eager)"),
                                                  ("compile", "disabled (enforce_eager)")]),
    (re.compile(r"cudagraph_mode[=:] ?'?(\w+)"), lambda m: [("cuda_graph.mode", m[1])]),
    (re.compile(r"Enabled custom fusions: (.+)$"), lambda m: [("compile.fusions", m[1].strip())]),
    # tqdm bars: "Capturing CUDA graphs (decode, FULL):   0%| 0/35"; one value per mode and size.
    (re.compile(r"Capturing CUDA graphs \(([^)]+)\):\s+\d+%\|[^|]*\|\s*\d+/(\d+)"),
     lambda m: [("cuda_graph.captures", f"{m[1]} x{m[2]}")]),
    (re.compile(r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)"),
     lambda m: [("cuda_graph.pool_gib", float(m[1]))]),
    (re.compile(r"FlashInfer top-p/top-k sampling (disabled|enabled)"),
     lambda m: [("backend.sampling_flashinfer", m[1])]),
    (re.compile(r"Using (V\d) Model Runner"), lambda m: [("model_runner", m[1])]),
    # V1 runner predates the "Using VN Model Runner" line; its identity is the
    # gpu_model_runner.py module (V2 uses model_runner.py).
    (re.compile(r"\[gpu_model_runner\.py:\d+\]"),
     lambda m: [("model_runner", "V1")]),
    # Parallel configuration: what TP/EP/PP the process actually runs with. This is
    # the only place TP>1 (and expert parallel) shows up in the startup log.
    (re.compile(r"world_size=(\d+) rank=(\d+) local_rank=(\d+)"),
     lambda m: [("parallel.world_size", int(m[1]))]),
    (re.compile(r"rank \d+ in world size \d+ is assigned as (DP rank \d+, PP rank \d+, PCP rank \d+, TP rank \d+(?:, ETP rank \S+)?, EP rank \S+, EPLB rank \S+)"),
     lambda m: [("parallel.roles", m[1])]),
    (re.compile(r"Using (\S+) KV cache layout"), lambda m: [("kv_cache.layout", m[1])]),
    # Model loading cost: memory footprint and wall time. Both V1 (gpu_model_runner)
    # and V2 (model_runner) print the same message.
    (re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"),
     lambda m: [("model_loading.gib", float(m[1])), ("model_loading.seconds", float(m[2]))]),
    # torch.compile outcome: artifact count/size reflects compile behaviour; total
    # wall time is comparable between versions on the same machine.
    (re.compile(r"collected artifacts: (.+?) total"),
     lambda m: [("compile.artifacts", m[1].strip())]),
    (re.compile(r"torch.compile took ([\d.]+) s in total"),
     lambda m: [("compile.total_s", float(m[1]))]),
    (re.compile(r"Pinned memory is not available.*falling back to (.+?) for UVA buffers"),
     lambda m: [("uva_buffers", f"fallback: {m[1]}")]),
    # Startup outcome. Without it, facts missing after a crash look like "not printed".
    (re.compile(r"Application startup complete"), lambda m: [("startup", "ready")]),
    (re.compile(r"EngineCore failed to start|Engine core initialization failed"),
     lambda m: [("startup", "failed")]),
    (re.compile(r"^(?:\(\w+ pid=\d+\) )?(?:ERROR .*?\] )?((?:RuntimeError|ValueError|torch\.OutOfMemoryError|AssertionError|AttributeError|KeyError|TypeError): .+)$"),
     lambda m: [("startup.error", m[1].strip())]),
    # SGLang.
    (re.compile(r"Disable prefill CUDA graph because"), lambda m: [("cuda_graph.prefill", "disabled")]),
    (re.compile(r"The server is fired up and ready to roll"), lambda m: [("startup", "ready")]),
    (re.compile(r"KV Cache is allocated\. dtype: (\S+?),"), lambda m: [("kv_cache.dtype", m[1])]),
]

# Arguments that mean the same thing in both engines, mapped to one name.
# value: (vLLM non-default arg, SGLang server_arg, transform for SGLang value)
SHARED_ARGS = {
    "model": ("model", "model_path", None),
    "dtype": ("dtype", "dtype", None),
    "max_model_len": ("max_model_len", "context_length", None),
    "gpu_memory_fraction": ("gpu_memory_utilization", "mem_fraction_static", None),
    "prefix_cache": ("enable_prefix_caching", "disable_radix_cache", lambda v: not v),
    "cuda_graph.disabled": ("enforce_eager", "disable_cuda_graph", None),
    "tensor_parallel_size": ("tensor_parallel_size", "tp_size", None),
    "kv_cache_dtype": ("kv_cache_dtype", "kv_cache_dtype", None),
    "backend.attention_arg": ("attention_backend", "attention_backend", None),
    "backend.moe_arg": ("moe_backend", "moe_runner_backend", None),
    "backend.sampling_arg": (None, "sampling_backend", None),
    "port": (None, "port", None),
}


def _literal_dict(text: str) -> dict | None:
    """Parse the `{...}` that starts `text`, tolerating trailing log text."""
    depth, quote = 0, None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote and text[i - 1] != "\\":
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
            if depth == 0:
                try:
                    value = ast.literal_eval(text[: i + 1])
                except (ValueError, SyntaxError):
                    return None
                return value if isinstance(value, dict) else None
    return None


def _call_kwargs(text: str) -> dict | None:
    """Parse SGLang's `ServerArgs(kw=v, ...)` call form into a dict.

    Newer SGLang prints `server_args=ServerArgs(model_path=..., tp_size=4, ...)`
    (a keyword-argument call, not a dict literal), so `ast.literal_eval` on the
    whole line fails and every shared arg is lost. Tolerate trailing log text.
    """
    match = re.match(r"([A-Za-z_]\w*)\(", text)
    if not match:
        return None
    depth, quote, end = 0, None, None
    for i in range(match.end(), len(text)):
        ch = text[i]
        if quote:
            if ch == quote and text[i - 1] != "\\":
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                end = i
                break
            depth -= 1
    if end is None:
        return None
    inner = text[match.end():end]
    # Split the keyword arguments at top level, then rebuild as a dict literal so
    # ast.literal_eval accepts it (the `dict(...)` call form is not supported).
    items = []
    parts, depth, quote, cur = [], 0, None, []
    for ch in inner:
        if quote:
            cur.append(ch)
            if ch == quote and (len(cur) < 2 or cur[-2] != "\\"):
                quote = None
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    for part in parts:
        if not part:
            continue
        eq = part.find("=")
        if eq < 1 or not part[:eq].strip().isidentifier():
            return None
        items.append('"%s": %s' % (part[:eq].strip(), part[eq + 1:].strip()))
    try:
        value = ast.literal_eval("{" + ", ".join(items) + "}")
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None


def parse(text: str) -> dict:
    facts: dict[str, Fact] = {}
    args: dict = {}
    engine = None

    def put(key, value, lineno, line):
        facts.setdefault(key, Fact()).add(value, lineno, line)

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = ANSI.sub("", raw)
        if "non-default args:" in line:
            engine = engine or "vllm"
            parsed = _literal_dict(line.split("non-default args:", 1)[1].strip())
            if parsed is not None:
                args.update(parsed)
                put("args.source", "vllm non-default args", lineno, line)
        elif "server_args=" in line:
            engine = engine or "sglang"
            tail = line.split("server_args=", 1)[1].strip()
            parsed = _literal_dict(tail) or _call_kwargs(tail)
            if parsed is not None:
                args.update(parsed)
                put("args.source", "sglang server_args (all args)", lineno, line)
        if engine is None and ("vllm" in line.lower() and "[" in line and ".py:" in line):
            engine = "vllm"
        for pattern, emit in LINE_RULES:
            match = pattern.search(line)
            if match:
                for key, value in emit(match):
                    # Errors after the server reported ready (e.g. during shutdown) say
                    # nothing about how it started.
                    if key == "startup.error":
                        if "ready" in (facts.get("startup") or Fact()).values:
                            key = "after_ready.error"
                        else:
                            put("startup", "failed", lineno, line)
                    put(key, value, lineno, line)

    if engine:
        facts.setdefault("engine", Fact()).add(engine, 0, "(detected from log format)")
    for name, (vllm_key, sglang_key, transform) in SHARED_ARGS.items():
        key = vllm_key if engine == "vllm" else sglang_key
        if key and key in args:
            value = args[key]
            if engine == "sglang" and transform:
                value = transform(value)
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            put(f"arg.{name}", value, 0, f"{'non-default args' if engine == 'vllm' else 'server_args'}[{key!r}]")
    return {"engine": engine, "facts": facts, "args": args}


def _shown(fact: Fact | None):
    return "<absent>" if fact is None else fact.value


def diff(a: dict, b: dict, raw_args: bool = False) -> list[dict]:
    rows = []
    for key in sorted(set(a["facts"]) | set(b["facts"])):
        fa, fb = a["facts"].get(key), b["facts"].get(key)
        va, vb = _shown(fa), _shown(fb)
        if fa is None or fb is None:
            status = "only_baseline" if fb is None else "only_candidate"
        else:
            status = "same" if va == vb else "changed"
        rows.append({"key": key, "baseline": va, "candidate": vb, "status": status,
                     "evidence": {"baseline": fa.evidence if fa else [],
                                  "candidate": fb.evidence if fb else []}})
    if raw_args and a["engine"] == b["engine"]:
        for key in sorted(set(a["args"]) | set(b["args"])):
            va, vb = a["args"].get(key, "<absent>"), b["args"].get(key, "<absent>")
            if va != vb:
                rows.append({"key": f"raw_arg.{key}", "baseline": va, "candidate": vb,
                             "status": "changed", "evidence": {}})
    return rows


def _fmt(value) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= 60 else text[:57] + "..."


def _md_cell(value) -> str:
    """Render a diff cell for a markdown table: escape pipes, collapse lists."""
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text.replace("|", "\\|").replace("\n", " ")


def _render_markdown(*, path_a: str, path_b: str, engine_a, engine_b,
                     rows: list[dict], warnings: list[str],
                     differing: int, identical: int, all_rows: bool) -> str:
    """Render a GitHub-flavored diff report, suitable for pasting into an issue."""
    lines: list[str] = ["# servediff report", ""]
    eng = f" (`{engine_a}` vs `{engine_b}`)" if engine_a and engine_b and engine_a != engine_b else ""
    lines.append(f"Comparing `{path_a}` vs `{path_b}`{eng}.")
    lines.append("")
    for w in warnings:
        lines.append(f"> ⚠️ {w}")
    if warnings:
        lines.append("")
    lines += ["| Fact | Baseline | Candidate | Status |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['key']}` | {_md_cell(r['baseline'])} | {_md_cell(r['candidate'])} | {r['status']} |")
    lines.append("")
    lines.append(f"**{differing} difference(s); {identical} fact(s) identical.** "
                 "Absent means the log did not print it, not that it was off."
                 + (" Shown: all facts." if all_rows else ""))
    lines.append("")

    # Collapsible evidence: only for rows that have it.
    ev_lines: list[str] = []
    for r in rows:
        for side, label in (("baseline", path_a), ("candidate", path_b)):
            ev = r.get("evidence", {}).get(side, [])
            for lineno, text in ev:
                ev_lines.append(f"- `{r['key']}` — {label} line {lineno}:")
                ev_lines.append(f"  > {text}")
    if ev_lines:
        lines.append("<details>")
        lines.append("<summary>Evidence</summary>")
        lines.append("")
        lines.extend(ev_lines)
        lines.append("")
        lines.append("</details>")
        lines.append("")
    lines.append("*Generated by [servediff](https://pypi.org/project/servediff/)*")
    return "\n".join(lines)


def _version() -> str:
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("servediff")
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"servediff {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", help="effective config facts in one log")
    show.add_argument("log", type=Path)
    show.add_argument("--json", action="store_true")
    d = sub.add_parser("diff", help="diff two server logs, or two `servediff env` snapshots")
    d.add_argument("baseline", type=Path)
    d.add_argument("candidate", type=Path)
    d.add_argument("--raw-args", action="store_true", help="also diff every raw argument (same engine only)")
    d.add_argument("--json", action="store_true")
    d.add_argument("--markdown", action="store_true", help="render a GitHub-flavored report (paste into issues)")
    d.add_argument("--all", action="store_true", help="also print unchanged facts")
    e = sub.add_parser("env", help="snapshot package versions, GPU and engine env vars of a server environment")
    e.add_argument("--python", help="interpreter of the server's environment (default: this one)")
    e.add_argument("-o", "--output", type=Path, help="write JSON here instead of stdout")
    c = sub.add_parser("capture", help="run a server command, tee its log, and snapshot its environment")
    c.add_argument("-o", "--output", type=Path, default=Path("servediff-capture.log"),
                   help="log file (default: servediff-capture.log)")
    c.add_argument("--python", help="interpreter for the env snapshot (default: this one)")
    c.add_argument("run_cmd", nargs=argparse.REMAINDER, metavar="-- command",
                   help="server command, after --")
    args = parser.parse_args(argv)

    if args.command == "capture":
        cmd = args.run_cmd
        if cmd[:1] == ["--"]:
            cmd = cmd[1:]
        if not cmd:
            print("capture: provide a command after --, e.g. servediff capture -- vllm serve <model>",
                  file=sys.stderr)
            return 2
        # Snapshot the environment before the server starts, so we capture what it launched with.
        try:
            snap = envsnap.snapshot(args.python)
        except (RuntimeError, OSError) as exc:
            print(f"servediff capture: env snapshot failed: {exc}", file=sys.stderr)
            return 2
        snap_path = args.output.with_suffix(args.output.suffix + ".env.json")
        snap_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                             encoding="utf-8")
        print(f"env snapshot -> {snap_path}", file=sys.stderr)
        print(f"log -> {args.output}  (running: {' '.join(shlex.quote(p) for p in cmd)})", file=sys.stderr)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
        except (FileNotFoundError, OSError) as exc:
            print(f"capture: cannot start {cmd[0]}: {exc}", file=sys.stderr)
            return 2
        with open(args.output, "w", encoding="utf-8", errors="replace") as logf:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()
        print(f"\nserver exited with code {proc.returncode}; log: {args.output}", file=sys.stderr)
        return proc.returncode

    if args.command == "env":
        try:
            snap = envsnap.snapshot(args.python)
        except (RuntimeError, OSError) as exc:
            print(f"servediff env: {exc}", file=sys.stderr)
            return 2
        text = json.dumps(snap, indent=2, ensure_ascii=False, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
            print(f"wrote {args.output}: {len(snap['packages'])} packages, {len(snap['env'])} env vars", file=sys.stderr)
        else:
            print(text)
        return 0

    if args.command == "show":
        parsed = parse(args.log.read_text(encoding="utf-8", errors="replace"))
        if not parsed["facts"]:
            print(f"{args.log}: no vLLM or SGLang configuration lines recognised", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps({k: {"value": f.value, "evidence": f.evidence} for k, f in parsed["facts"].items()},
                             indent=2, ensure_ascii=False))
        else:
            for key, fact in sorted(parsed["facts"].items()):
                where = fact.evidence[0][0]
                print(f"{key:32s} {_fmt(fact.value):60s} line {where if where else '-'}")
        return 0

    text_a = args.baseline.read_text(encoding="utf-8", errors="replace")
    text_b = args.candidate.read_text(encoding="utf-8", errors="replace")
    snaps = envsnap.is_snapshot(text_a), envsnap.is_snapshot(text_b)
    if any(snaps):
        if not all(snaps):
            print("diff: give two server logs or two `servediff env` snapshots, not one of each", file=sys.stderr)
            return 2
        rows = envsnap.ranked(envsnap.diff(json.loads(text_a), json.loads(text_b)))
        key = [r for r in rows if envsnap.is_key(r)]
        if args.json:
            print(json.dumps({"differences": len(rows), "rows": rows}, indent=2, ensure_ascii=False))
        elif args.markdown:
            shown = rows if args.all else key
            print(_render_markdown(
                path_a=str(args.baseline), path_b=str(args.candidate),
                engine_a=None, engine_b=None,
                rows=shown, warnings=[],
                differing=len(key), identical=len(rows) - len(key),
                all_rows=bool(args.all)))
        else:
            shown = rows if args.all else key
            print(f"{'fact':32s} {'baseline':40s} {'candidate':40s} status")
            for r in shown:
                print(f"{r['key']:32s} {_fmt(r['baseline'])[:40]:40s} {_fmt(r['candidate'])[:40]:40s} {r['status']}")
            print(f"\n{len(key)} difference(s) in Python, GPU, inference packages and engine env vars;"
                  f" {len(rows) - len(key)} in other packages{'' if args.all else ' (--all to list them)'}.")
        return 1 if rows else 0

    a = parse(text_a)
    b = parse(text_b)
    for path, parsed in ((args.baseline, a), (args.candidate, b)):
        if not parsed["facts"]:
            print(f"{path}: no vLLM or SGLang configuration lines recognised", file=sys.stderr)
            return 2
    rows = diff(a, b, raw_args=args.raw_args)
    if not args.all:  # errors after "ready" (e.g. on shutdown) say nothing about the config
        rows = [r for r in rows if not r["key"].startswith("after_ready.")]
    differing = [r for r in rows if r["status"] != "same"]
    if args.json:
        print(json.dumps({"differences": len(differing), "rows": rows}, indent=2, ensure_ascii=False, default=str))
    elif args.markdown:
        warnings = []
        for side, parsed in (("baseline", a), ("candidate", b)):
            if "failed" in (parsed["facts"].get("startup") or Fact()).values:
                error = parsed["facts"].get("startup.error")
                msg = f"{side} server failed to start"
                if error:
                    msg += f": {_fmt(error.values[0])}"
                msg += "; facts after the failure are absent because startup stopped, not because they were off"
                warnings.append(msg)
        shown_rows = rows if args.all else differing
        print(_render_markdown(
            path_a=str(args.baseline), path_b=str(args.candidate),
            engine_a=a["engine"], engine_b=b["engine"],
            rows=shown_rows, warnings=warnings,
            differing=len(differing), identical=len(rows) - len(differing),
            all_rows=bool(args.all)))
    else:
        if a["engine"] != b["engine"]:
            print(f"note: comparing {a['engine']} with {b['engine']}; only shared facts line up\n")
        for side, parsed in (("baseline", a), ("candidate", b)):
            if "failed" in (parsed["facts"].get("startup") or Fact()).values:
                error = parsed["facts"].get("startup.error")
                print(f"warning: {side} server failed to start"
                      f"{': ' + _fmt(error.values[0]) if error else ''}; facts after the failure"  # first = root cause
                      " are absent because startup stopped, not because they were off\n")
        print(f"{'fact':32s} {'baseline':40s} {'candidate':40s} status")
        for r in rows if args.all else differing:
            print(f"{r['key']:32s} {_fmt(r['baseline'])[:40]:40s} {_fmt(r['candidate'])[:40]:40s} {r['status']}")
        print(f"\n{len(differing)} difference(s); {len(rows) - len(differing)} fact(s) identical."
              " Absent means the log did not print it, not that it was off.")
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
