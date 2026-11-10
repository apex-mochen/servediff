"""Which log lines servediff reads nothing from.

A tool that claims to show "what the server actually ran with" is defined as much by
the lines it ignores as by the ones it reads. This report is that gap: for each log,
every line no rule matched, deduplicated by its shape (digits replaced with N), with
counts. It is what an external user should paste into an issue when a fact comes out
`<absent>` and they suspect the line was there.

Most of that gap is noise — route tables, ASCII banners, deprecation warnings, progress
bars, access logs — which no fact could ever come from. Flooding the report with it is
how a report stops being read, so noise is counted and grouped separately from the
shapes that might actually be a missed configuration line.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from servediff.cli import ANSI, BRACKET_TS, LINE_RULES, parse

DIGITS = re.compile(r"\d+")

# A real multi-process log prefixes every line with who printed it and from where:
# ``(Worker_TP8 pid=123) INFO 11-06 12:00:00 [cuda.py:118] ...``. The noise patterns below
# describe the *message*, so they are tested against the message too, not only against a
# bare line — otherwise a whole traceback reads as "worth a look" just because each frame
# carries a `(EngineCore pid=…) ERROR … [core.py:…]` prefix. Only a prefix is ever stripped;
# the message itself is passed through untouched.
_PREFIX = (
    re.compile(r"\([^)]*pid=\d+\)\s*"),
    re.compile(r"(?:DEBUG|INFO|WARNING|ERROR|CRITICAL|VERBOSE)\b[:\s-]*"),
    # vLLM prints `INFO 10-28 11:26:34` — month-day, no year.
    re.compile(r"(?:\d{4}-)?\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:[,.]\d{3})?\s*"),
    re.compile(r"^-+\s+"),
    re.compile(r"^\[?[\w./]+\.py:\d+\]?\s*-?\s*"),
)


def message(line: str) -> str:
    """The line with its process/time/module preamble removed."""
    text = BRACKET_TS.sub("", ANSI.sub("", line)).strip()
    for _ in range(12):
        before = text
        for pattern in _PREFIX:
            if pattern.match(text):
                text = pattern.sub("", text, count=1).strip()
        if text == before:
            break
    return text


# Lines whose content is never configuration. Each pattern must be narrow enough that a
# real config line cannot match it: widening one of these silently hides a missed fact.
# They are matched against the *raw* line, not the shape, because collapsing digits
# destroys the input they key on (a 200 status becomes N).
NOISE = (
    # An InfiniBand NIC topology table (GB200 / HGX multi-node): the same grid as the
    # GPU one, one row per NIC, its chip names, and its legend.
    re.compile(r"^NIC\s*\d+: \S+$"),
    re.compile(r"^NIC\d+([\s]|$)"),
    re.compile(r"^NIC Legend:"),
    # `ldd` output: one absolute path per shared library, a dependency dump rather
    # than a runtime setting.
    re.compile(r"^/[\w./+-]+\.so[\w.]*$"),
    re.compile(r"Route: /\S*"),  # the HTTP route table
    re.compile(r"[\u2580-\u259f]{2,}"),  # box-drawing banners (██▄█▀)
    re.compile(r"\[\w+\.py:\d+\]\s*$"),  # banner lines that printed nothing after the tag
    re.compile(r"warnings\.warn\("),
    re.compile(r"DeprecationWarning|is deprecated, use"),
    re.compile(r"Loading (safetensors )?checkpoint shards"),  # tqdm progress bar
    re.compile(r'"(GET|POST) /[^"]*" \d{3}'),  # uvicorn access log
    re.compile(r"INFO:\s+(Started server process|Waiting for application startup)"),
    re.compile(r"Available routes are:"),
    # Python tracebacks: the root cause already lands in startup.error, the frames
    # themselves are never configuration.
    re.compile(r"^Traceback \(most recent call last\)"),
    re.compile(r'^\s*File "[^"]*", line \d+'),
    re.compile(r"^\s*[\^~]{3,}\s*$"),
    re.compile(r"^During handling of the above exception|^Process \w+:|^Call stack:"),
    # The source lines a traceback echoes; each is part of a call, not a setting.
    re.compile(r"^\s*(self\.|return |raise |yield |assert |del |super\(\)|with |async with"
               r"|await |next\(|import |from )"),
    re.compile(r"^\s*(?:[\w.]+(?:,\s*[\w.]+)*|[\w.]+:\s*[\w.\[\]]+)\s*=\s*(?:await\s+)?[\w.]+\s*\("),  # `x = Y(` frame echo
    re.compile(r"^\s*[\w.]+\([^()]*(?:\([^()]*\)[^()]*)*\)(?:\))?\s*$"),  # `uvloop.run(run_server(args))`
    re.compile(r"^\s*[\w.]+\($"),  # a call split across lines: `super().__init__(`
    # A logging failure's own dump: the traceback is about logging, not about the server.
    re.compile(r"^(--- )?Logging error ---"),
    re.compile(r"^Message: |^Arguments: "),
    # Engine teardown, printed on every shutdown, with the settings of nothing.
    re.compile(r"\[shutdown\]|\[shutdown\]|trigger received signal=SIGTERM"),
    # Phase markers: they say a step began, never with what settings it ran.
    re.compile(r"JIT kernel warmup starting\.|Starting torch\.compile$"),
    # A per-machine absolute path, so it can never be equal across two boxes and no
    # setting hides inside it.
    re.compile(r"Using cache directory: |saved AOT compiled function to "),
    # Serving telemetry, printed while the server runs: a metric, not a startup setting.
    re.compile(r"Avg prompt throughput"),
    # Inter-process plumbing warnings and resource teardown.
    re.compile(r"resource_tracker: There appear to be"),
    re.compile(r"destroy_process_group\(\) was not called"),
    re.compile(r"No available shared memory broadcast block found"),
    re.compile(r"Guessing device ID based on global rank"),
    re.compile(r"The client socket cannot be initialized to connect to"),
    # The traceback's own terminator, which carries no message of its own.
    re.compile(r"^(SystemExit|KeyboardInterrupt)$"),
    # uvicorn's own shutdown narration.
    re.compile(r"^(Shutting down|Waiting for application shutdown"
               r"|Application shutdown complete|Started server process)\.?$"),
    # Text pasted from a bug-report template rather than printed by the server.
    re.compile(r"^#{2,} |^- \[[ x]\] "),
    # Framework diagnostics that name their own source file before saying anything.
    re.compile(r"^\S+\.py:\d+: (User)?Warning:"),
    re.compile(r"\[Autotuner\]: Autotuning process starts"),
    # A wrapper script's own output, not the server's: the benchmark that launched vLLM
    # prints which config it is running. Auditing those lines finds no rule that could
    # ever read them, because they are not vLLM log lines.
    re.compile(r"^>>> |^\[spec_bench\]|^!!! "),
    # A bare `AssertionError` under a failed import check: the detail is the line above,
    # which is read as `kernel.module_missing` / `kernel.missing_lib`.
    re.compile(r"^AssertionError$"),
    re.compile(r"^I want to run inference of a \[specific model\]"),
    # `collect_env`, the block every bug report pastes. On ROCm it is ~180 lines and every
    # one of them reads like a setting, so it drowned the report. The version lines inside
    # it are facts (`env.*`); its scaffolding and machine inventory are not.
    re.compile(r"^Collecting environment information\.\.\.$"),
    re.compile(r"^=+$"),
    re.compile(r"^\s*(System Info|PyTorch Info|Python Environment|CUDA / GPU Info"
               r"|CPU Info|Versions of relevant libraries|vLLM Info|Environment Variables"
               r"|CPU)\s*:?\s*$"),
    re.compile(r"^Vulnerability .+:"),  # CPU mitigations: per-host, not a setting
    re.compile(r"^Flags:"),
    re.compile(r"^\s+(?:[a-z0-9_.+-]+ ){3,}"),  # the CPU flag dump wrapping onto more lines
    # The same dump continued at column 0: nothing but lowercase CPU-feature tokens. A real
    # setting line always carries a separator or a value after a colon.
    re.compile(r"^(?:[a-z0-9_][a-z0-9_.+-]* ){4,}[a-z0-9_.+-]*$"),
    # A container runtime's device-capability list: `brand=tesla,driver>=470` repeated across
    # wrapped lines. It describes the driver's defaults, not the run.
    re.compile(r"^(?=.*brand=).*driver"),
    re.compile(r"^(?:GCC|Clang|Libc|Python platform)\s+version\s*:"
               r"|^Python platform\s*:"),
    re.compile(r"^(?:GCC|Clang|Libc) version\s*:"),
    re.compile(r"^CMake version\s*:"),
    re.compile(r"^(?:CUDA|MIOpen) runtime version\s*:"),
    re.compile(r"^vLLM Build Flags:"),
    re.compile(r"^\s*CUDA Archs: "),
    re.compile(r"^Could not collect$"),
    re.compile(r"^\[pip3\] (?:conch-triton-kernels|onnx|onnx-ir|onnxscript|onnxslim|pyzmq"
               r"|torchaudio|torchvision|onnxruntime)=="),
    re.compile(r"^(?:Architecture|CPU op-mode\(s\)|Address sizes|Byte Order|CPU\(s\)"
               r"|On-line CPU\(s\) list|Vendor ID|Model name|CPU family|Model|Thread\(s\) per core"
               r"|Core\(s\) per socket|Socket\(s\)|Stepping|Frequency boost|CPU\(s\) scaling MHz"
               r"|CPU max MHz|CPU min MHz|BogoMIPS|Virtualization|NUMA node\(s\)"
               r"|NUMA node\d+ CPU\(s\)|L1d cache|L1i cache|L2 cache|L3 cache)\s*:"),
    re.compile(r"^(?:Is debug build|CUDA used to build PyTorch|XPU used to build PyTorch"
               r"|CUDA_MODULE_LOADING set to|Nvidia driver version|cuDNN version|MIOpen runtime"
               r"|Is XNNPACK available|Is CUDA available|Is XPU available|GPU Topology)\s*:"),
    re.compile(r"^\s+(?:Could not collect|CUDA Archs:)"),
    # One wheel per GPU architecture, so this list is a build artifact, not a setting; on a
    # ROCm box it is ~40 identical lines.
    re.compile(r"^\[pip3\] amd-(?:torch|torchvision)-device-"),
    re.compile(r"^\[conda\] Could not collect"),
    # The launch command pasted into a bug report, wrapped with `\` continuations. It is the
    # user's shell history, not output: the server prints its own args as "non-default args".
    re.compile(r"^\s*--\S+(?: \S+)* ?\\?$"),
    re.compile(r"^(?:vllm|python3?|sglang) [^\n]*\\$"),
    # The tail of a wrapped LD_LIBRARY_PATH value, which the env-var rule already reads.
    re.compile(r"^[a-z0-9_.-]+/(?:lib|site-packages)"),
    # The pip inventory: one line per installed wheel. A CUDA container lists ~35 of them
    # and only a few decide kernels, which the [pip3] rule already reads.
    re.compile(r"^\[pip3\] \S+=="),
    re.compile(r"^=+[^=]*=+$"),
    # GPU topology and rocm-smi print the machine, not the run: the grid, its legend, the
    # Numa Nodes table, and the weight / hops / link-type matrices an 8-GPU box emits.
    re.compile(r"^GPU\d+[\t ]"),
    re.compile(r"^GPU\[\d+\]\s*.{0,3}:\s*\(Topology\)"),
    re.compile(r"^GPU \d+: "),
    # A multi-GPU box lists them on one line, `GPU 0,1,2,3,4,5,6,7: NVIDIA H20`, and its
    # capability row the same way.
    re.compile(r"^GPU [\d,]+: |^GPU [\d,]+ Compute Capability: "),
    # The topology matrix header: tab-separated column names, never a run's output.
    re.compile(r"^(?=.*CPU Affinity).*\t"),
    # A reporter's note inserted into the paste, the whole line one parenthetical.
    re.compile(r"^\([^()]*\)$"),
    re.compile(r"^Legend:"),
    re.compile(r"^[A-Z#]{1,4}\s+=\s"),
    re.compile(r"^(?:Hypervisor vendor|Virtualization type|Hypervisor)\s*:"),
    re.compile(r"^uv is set$"),
    # A variable printed with its name but no value: present, unset.
    re.compile(r"^[A-Z][A-Z0-9_]{2,}=$"),
    re.compile(r"^(?:CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|VLLM_CACHE_ROOT)="),
    # A report written around the logs: Markdown bullets, bold sentences, and the wrapper
    # lines that never appear in engine output.
    re.compile(r"^-\s+\S|\*\*"),
    re.compile(r"^`"),
    re.compile(r"^(?:I have tried|I tried|Steps to reproduce|Expected behavior"
               r"|Actual behavior|Additional context|Environment)\s*:"),
    re.compile(r"^[A-Z][a-z']+\s[^:=]{40,}[.]$"),
    # SGLang's own housekeeping: package inventory lines (`name: Module Not Found`), per-NIC
    # staging buffers with their addresses, NIXL agent chatter, and the kill sequence.
    re.compile(r"^[a-z][a-z0-9_.+-]*: (?:Module Not Found|\d)"),
    re.compile(r"^StagingBuffer allocated: |^Staging buffer using cudaMalloc"),
    re.compile(r"^NIXL INFO|^Initialized NIXL agent|^Backend UCX was instantiated"),
    re.compile(r"^Received sigquit from a child process|^kill_process_tree called"),
    re.compile(r"^CUDA available: (True|False)"),
    re.compile(r"^GPU \d+ Compute Capability: "),
    re.compile(r"^NVIDIA Topology:|^\t"),
    # SGLang's macOS report: the machine, not the run. An M-series box decides kernels by its
    # chip and Metal version, which are properties of the box, like the CPU flag dump.
    re.compile(r"^(?:MPS available|macOS Version|macOS Build|Apple Silicon|Unified Memory"
               r"|CPU Cores \([^)]*\)|Metal Support|GPU Cores)\s*:"),
    # Source code echoed around the logs: a Triton patch diff, a frame's assignment, a bare
    # closing bracket. A log line states `name: value` or a sentence, never this.
    re.compile(r"^\+\s|^[)\]}{]"),
    re.compile(r"^\s*[a-z_][\w.]*\s*="),
    # A bare attribute expression (`tensor.offset`), a frame echo whose name is upper case
    # (`WTYPE = scalar_types.uint8b128`), and a call left open across lines. A log line never
    # ends in `(` and never states an expression with nothing to say about it.
    re.compile(r"^[a-z_][\w]*(?:\.[A-Za-z_]\w*)+$"),
    re.compile(r"^[A-Z][A-Z0-9_]+ (?:=|\+=) "),
    re.compile(r"^\s*\S.*\($"),
    # `\b` sits after the keyword, not after the space: a loop or branch whose condition is a
    # quoted string (`if "W4A4" in name:`) put a non-word character right after the space, so
    # the boundary test failed and the line stayed in the report.
    re.compile(r"^\s*(?:if|elif|else|for|while|try|except)\b"),
    re.compile(r"^\.\.\.$"),
    # A reproducer script pasted into an issue body, which is not a log at all: a function
    # definition, a module-level tuple unpack, a subscripted assignment, a print. A log line
    # states `name: value` or a sentence, and never starts with one of these.
    re.compile(r"^(?:def|class)\s+[A-Za-z_]\w*[(:]"),
    re.compile(r"^[A-Za-z_]\w*(?:\[[^\]]*\])?(?:, [A-Za-z_]\w*(?:\[[^\]]*\])?)+ (?:=|\+=) "),
    re.compile(r"^[A-Za-z_][\w.]*\[[^\]]*\] (?:=|\+=|\() "),
    re.compile(r"^\s*print\("),
    # The rest of the same script: a section comment, a bare loop keyword, a string literal
    # standing alone, and a call continued on the next line (a log line never ends in a comma).
    re.compile(r"^#"),
    re.compile(r"^(?:continue|pass|break)$"),
    re.compile(r"^[A-Za-z_]\w*\s+\+= "),
    re.compile(r"^\s*f?[\"']"),
    re.compile(r"^\s*.*,\s*$"),
    # The other half of a call split across lines: a bare argument list
    # (`8, 96, 128, conv_kernel_size=4, num_spec=5`, `groups, KVCacheLayout.BLHNC`) and a
    # continuation that closes a parenthesis without ever starting a statement
    # (`…routing_input_mode=…), wp))(ml)`). A log line is a statement -- it starts with a word
    # and says something about it -- so it never consists of arguments alone. The first token
    # is required to be a bare name, which is what keeps `CUDA_VISIBLE_DEVICES=0,1,2,3` and
    # `NVIDIA_DRIVER_CAPABILITIES=compute,utility` out: those are facts, and the env-var rule
    # reads them.
    re.compile(r"^\s*[A-Za-z_0-9][\w.]*(?:,\s*(?:[A-Za-z_0-9][\w.]*"
               r"|[A-Za-z_][\w.]*=[^\s,)]+))+\s*$"),
    re.compile(r"^(?=.*,\s*[A-Za-z_]\w*=)(?=.*\)\)).*\)\s*$"),
    # A conditional expression standing alone (`run(a) if a.cmd == "run" else compare(a)`):
    # it picks between two values, so there is no subject for it to say anything about. The
    # line is required to *begin* with a call, which is what keeps a warning that merely
    # contains both keywords (`… if available else falling back … (set X=0 to disable)`) out of
    # the noise table. The wider `if … else … )` shape was measured first and rejected: it ate
    # exactly that warning.
    re.compile(r"^[A-Za-z_][\w.]*\([^()]*\)\s+if\s+.+\s+else\s+.+\)$"),
    # A month later, `collect_env` brought its whole package inventory with it: four reports
    # carried ~41 `[conda]` rows each, and every row reads like a version fact. The `[pip3]`
    # form above is already read for the few wheels that decide kernels; the `[conda]` table is
    # the same inventory one column-format away, so the same policy applies. `[conda] Could not
    # collect` was already here; this covers the rows.
    re.compile(r"^\[conda\] \S+"),
    # A pasted reproducer's *expression* lines, which the statement-shaped rules above still
    # miss: an f-string (a real log line never carries `f"…{…}"`), a tuple/dict entry built
    # inside a call, a nested call closed twice on one line, a bare name that only closes a
    # parenthesis, a lone `[`, an annotation-only assignment, a comparison whose left operand
    # was on the previous line, and the continuation of a boolean condition.
    re.compile(r"\bf[\"']\{"),
    re.compile(r"^(?:async\s+(?:def|with|for)\b|async\s+def\b)"),
    re.compile(r"^[A-Za-z_]\w*:\s*[A-Za-z_][\w\[\], ']*=\s"),
    re.compile(r"^!=\s*\("),
    re.compile(r"^\s*\[\s*$"),
    re.compile(r"^\s*[A-Za-z_][\w.]*\)\s*$"),
    re.compile(r"^\s*[A-Za-z_][\w.]+\(.*\)\)\s*$"),
    re.compile(r"^(?:and|or)\b.*\)\s*$"),
    # The two rules above, widened by the dotted attribute names the same scripts use:
    # `layer.output_size_per_partition, layer.input_size_per_partition = N, K` is a tuple
    # assignment with qualified names, and `layer.weight_global_scale, layer.workspace, N, K)`
    # is the argument list that closes the call it came from.
    re.compile(r"^[A-Za-z_][\w.]*(?:, [A-Za-z_][\w.]*)+ (?:=|\+=) "),
    re.compile(r"^\s*[A-Za-z_0-9][\w.]*(?:,\s*(?:[A-Za-z_0-9][\w.]*"
               r"|[A-Za-z_][\w.]*=[^\s,)]+))+\s*\)\s*$"),
    # A parenthesized expression whose value is then used by attribute
    # (`(r / 'pressure-native-inputs.json').write_text(…)`). The wider
    # `^\(.*\)$` was measured first and rejected: it ate 31 real vLLM lines,
    # every one of them a `(EngineCore pid=…)` log entry.
    re.compile(r"^\s*\(.*\)\.[A-Za-z_]"),
    # The last three script shapes this batch brought: a tuple of layer specs whose line ends
    # the `for … in (` header it belongs to, a one-line call chain
    # (`tf.AutoModel….from_config(cfg).to(…).save_pretrained(path)`), and a `try` block's
    # `finally:`, which the keyword rule above lists every sibling of except this one.
    re.compile(r"^\s*\(.*\)\s*:\s*$"),
    re.compile(r"^\s*[A-Za-z_][\w.]*\(.*\)(?:\.\w+\(.*\))+\s*$"),
    re.compile(r"^\s*finally\s*:"),
)

# A line that is nothing but the prefix: `(EngineCore pid=3811)`.
_BARE_PREFIX = re.compile(r"^\s*\([^)]*pid=\d+\)\s*$")
_COMMENT = re.compile(r"\s+#\s.*$")


def shape(line: str) -> str:
    """The line with numbers collapsed, so progress bars and ports group together."""
    return DIGITS.sub("N", BRACKET_TS.sub("", ANSI.sub("", line)).strip())


def unread(text: str) -> Counter:
    """Count the lines no rule reads, keyed by shape.

    Applies the same normalisation parse() does before matching, so a line appears here
    only when the parser really reads nothing from it.
    """
    return unread_samples(text)[0]


def unread_samples(text: str) -> tuple[Counter, dict[str, str]]:
    """(counts by shape, one raw line per shape), for classifying without losing digits."""
    counts: Counter = Counter()
    samples: dict[str, str] = {}
    in_doc = False
    for raw in text.splitlines():
        line = BRACKET_TS.sub("", ANSI.sub("", raw))
        stripped = line.strip()
        # A docstring echoed by a pasted script: prose between two `"""`, which no log line
        # ever carries, so it is noise rather than a setting this tool failed to read.
        if in_doc:
            in_doc = stripped.count('"""') % 2 == 0
            continue
        if stripped.startswith('"""') or stripped.endswith('"""'):
            in_doc = stripped.count('"""') % 2 == 1
            continue
        if not line.strip():
            continue
        if "non-default args:" in line or "server_args=" in line:
            continue  # consumed as argument dicts, not by line rules
        if any(pattern.search(line) for pattern, _ in LINE_RULES):
            continue
        key = shape(line)
        counts[key] += 1
        samples.setdefault(key, line.strip())
    return counts, samples


def is_noise(raw_line: str) -> bool:
    if _BARE_PREFIX.match(raw_line):
        return True
    if any(pattern.search(raw_line) for pattern in NOISE):
        return True
    # The message without its preamble, and again with a trailing `# comment` gone: a
    # traceback echoes sources with comments in them (`_find_cuda_home()  # CUDA home`).
    stripped = message(raw_line)
    for candidate in (stripped, _COMMENT.sub("", stripped)):
        if candidate and any(pattern.search(candidate) for pattern in NOISE):
            return True
    return False


def split(counts: Counter, samples: dict[str, str]) -> tuple[Counter, Counter]:
    """(shapes worth a look, noise), both preserving their counts."""
    noise: Counter = Counter()
    look: Counter = Counter()
    for line, count in counts.items():
        (noise if is_noise(samples.get(line, line)) else look)[line] = count
    return look, noise


def _rows(lines: list[str], counts: Counter) -> None:
    for line, count in counts.most_common()[:200]:
        lines.append(f"  {count:>4}  {line[:150]}")


def report(path: Path, text: str, limit: int = 0, everything: bool = False) -> str:
    """One block per log: totals first, then shapes, noise separated out."""
    counts, samples = unread_samples(text)
    lines = [f"{path}: {sum(counts.values())} unread line(s), {len(counts)} shape(s)"]
    if not counts:
        lines.append("  nothing unread: every non-empty line matched a rule")
        return "\n".join(lines)
    look, noise = split(counts, samples)
    if look:
        lines.append(f"  {sum(look.values())} line(s) worth a look — no rule reads them and"
                     " they are not routes, banners, deprecations, progress bars or access logs:")
        rows = look.most_common(limit or None)
        for line, count in rows:
            lines.append(f"  {count:>4}  {line[:150]}")
    else:
        lines.append("  nothing worth a look: every unread line is noise.")
    if noise and (everything or not look):
        lines.append(f"  {sum(noise.values())} line(s) of noise"
                     f" (routes, banners, deprecations, progress bars, access logs):")
        _rows(lines, noise)
    elif noise:
        lines.append(f"  {sum(noise.values())} line(s) of noise (routes, banners, deprecations,"
                     " progress bars, access logs); --everything to list them")
    return "\n".join(lines)
