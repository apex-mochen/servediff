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
# SGLang wraps its timestamp in brackets: `[2026-10-06 00:15:19] Tree cache …`.
BRACKET_TS = re.compile(r"^\[(?:\d{4}-)?\d\d-\d\d \d\d:\d\d:\d\d\]\s*")

# `Could not collect`, `N/A` and `Not Set` are absences, not values: read literally they
# produced a fact like env.rocm = "Could" on a CUDA box, which then matched itself.
_ABSENT = ("n/a", "not set", "unknown")


def _env(key, value):
    text = str(value).strip()
    if not text or text.lower() in _ABSENT or text.lower().startswith("could not collect"):
        return []
    return [(key, text)]


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
    (re.compile(r"^(?!CMake)\W?[^\n]*\bversion (\d+\.\d+\.\d+\S*)"), lambda m: [("engine.version", m[1])]),
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
    # The backends this build would consider at all, printed on its own line. A diff
    # that shows the chosen backend unchanged can still hide this list shrinking.
    (re.compile(r"^backends: (\[[^\]]*\])"), lambda m: [("backend.candidates", m[1])]),
    (re.compile(r"Enforce eager set"), lambda m: [("cuda_graph", "disabled (enforce_eager)"),
                                                  ("compile", "disabled (enforce_eager)")]),
    (re.compile(r"cudagraph_mode[=:] ?'?(\w+)"), lambda m: [("cuda_graph.mode", m[1])]),
    (re.compile(r"Enabled custom fusions: (.+)$"), lambda m: [("compile.fusions", m[1].strip())]),
    # tqdm bars: "Capturing CUDA graphs (decode, FULL):   0%| 0/35"; one value per mode and size.
    (re.compile(r"Capturing CUDA graphs \(([^)]+)\):\s+\d+%\|[^|]*\|\s*\d+/(\d+)"),
     lambda m: [("cuda_graph.captures", f"{m[1]} x{m[2]}")]),
    (re.compile(r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)"),
     lambda m: [("cuda_graph.pool_gib", float(m[1]))]),
    # The end of a capture pass: "Graph capturing finished in 3 secs, took 0.25 GiB".
    # Printed once per pass (decode and prefill capture separately), so both values are
    # kept -- a capture that took twice as long is a startup-time regression on its own.
    (re.compile(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB"),
     lambda m: [("cuda_graph.capture_s", int(m[1])), ("cuda_graph.capture_gib", float(m[2]))]),
    (re.compile(r"FlashInfer top-p/top-k sampling (disabled|enabled)"),
     lambda m: [("backend.sampling_flashinfer", m[1])]),
    (re.compile(r"Using (V\d) Model Runner"), lambda m: [("model_runner", m[1])]),
    # V1 runner predates the "Using VN Model Runner" line; its identity is the
    # gpu_model_runner.py module (V2 uses model_runner.py).
    (re.compile(r"\[gpu_model_runner\.py:\d+\]"),
     lambda m: [("model_runner", "V1")]),
    # All-reduce is chosen from a candidate list and the chosen list is printed
    # per TP group; this is the only place the communication path (custom / NCCL /
    # FlashInfer) shows up in a startup log.
    (re.compile(r"Using (\[[^\]]*\]) all-reduce backends (?:\(in dispatch order\) )?for group '([^']+)' out of potential backends: (\[[^\]]*\])"),
     lambda m: [("backend.allreduce_dispatch", f"{m[2]}: {m[1]}"),
                ("backend.allreduce_potential", m[3])]),
    # Same statement when the line was truncated before the potential list, which
    # log sinks do on long lines. Anchored to a real log prefix (`[module.py:NN]`)
    # so a sentence quoted in a pasted issue body does not become a fact.
    (re.compile(r"\[(?:\w+\.py):\d+\][^\n]*?Using (\[[^\]]*\]) all-reduce backends (?:\(in dispatch order\) )?for group '([^']+)'"),
     lambda m: [("backend.allreduce_dispatch", f"{m[2]}: {m[1]}")]),
    (re.compile(r"Auto-selected flashinfer allreduce backend: (\w+)"),
     lambda m: [("backend.allreduce_flashinfer", m[1])]),
    # FlashInfer builds these only when its all-reduce/norm fusion path is active,
    # so their presence is the positive evidence that the fused path ran.
    (re.compile(r"Initialized FlashInfer Allreduce (norm quantization|norm) fusion workspace with backend=(\w+)"),
     lambda m: [("backend.allreduce_fusion", f"{m[1]}: {m[2]}")]),
    # Parallel configuration: what TP/EP/PP the process actually runs with. This is
    # the only place TP>1 (and expert parallel) shows up in the startup log.
    (re.compile(r"world_size=(\d+) rank=(\d+) local_rank=(\d+)"),
     lambda m: [("parallel.world_size", int(m[1]))]),
    (re.compile(r"rank \d+ in world size \d+ is assigned as (DP rank \d+, PP rank \d+, PCP rank \d+, TP rank \d+(?:, ETP rank \S+)?, EP rank \S+, EPLB rank \S+)"),
     lambda m: [("parallel.roles", m[1])]),
    (re.compile(r"Using (\S+) KV cache layout"), lambda m: [("kv_cache.layout", m[1])]),
    # Model identity as the engine resolved it, and the context length it settled on.
    # Both can differ from the requested args (auto-derivation, --max-model-len caps),
    # so a diff that only shows args moving does not explain the run.
    (re.compile(r"Resolved architecture: (\S+)"), lambda m: [("model.architecture", m[1])]),
    (re.compile(r"Using max model len (\d+)"), lambda m: [("model.max_len", int(m[1]))]),
    # KV-cache capacity after memory profiling: the numbers people watch when
    # concurrency drops after an upgrade.
    (re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
     lambda m: [("kv_cache.memory_gib", float(m[1]))]),
    (re.compile(r"GPU KV cache size: (\d[\d,]*) tokens"),
     lambda m: [("kv_cache.tokens", int(m[1].replace(",", "")))]),
    (re.compile(r"Maximum concurrency for (\d[\d,]*) tokens per request: ([\d.]+)x"),
     lambda m: [("kv_cache.max_concurrency", float(m[2]))]),
    # Engine init covers profiling, KV-cache allocation and warmup; the parenthesised
    # part is torch.compile alone, recorded separately as compile.total_s.
    (re.compile(r"init engine .*? took ([\d.]+) s"), lambda m: [("engine.init_s", float(m[1]))]),
    (re.compile(r"Starting vLLM server on (http://\S+)"), lambda m: [("server.bind", m[1])]),
    # `collect_env` — the block vLLM asks every bug report to paste. On ROCm it is
    # hundreds of lines, and each one reads like a setting, so the audit used to call all of
    # them "worth a look". The version lines are the facts that matter: a torch built
    # against a different ROCm changes every kernel a run can pick.
    (re.compile(r"^vLLM Version\s+:\s*(\S+)(?: \(git sha: (\w+), date: (\w+)\))?"),
     lambda m: [("engine.version", m[1]), ("engine.git_sha", m[2])] if m[2] else [("engine.version", m[1])]),
    (re.compile(r"^PyTorch version\s+:\s*(\S+)"), lambda m: _env("env.torch", m[1])),
    (re.compile(r"^ROCM used to build PyTorch\s+:\s*(\S+)"), lambda m: _env("env.torch_rocm", m[1])),
    (re.compile(r"^ROCM Version\s+:\s*(.+)$"), lambda m: _env("env.rocm", m[1])),
    (re.compile(r"^HIP runtime version\s+:\s*(.+)$"), lambda m: _env("env.hip", m[1])),
    (re.compile(r"^OS\s+:\s*(.+)$"), lambda m: [("env.os", m[1].strip())]),
    (re.compile(r"^Python version\s+:\s*(\S+)"), lambda m: [("env.python", m[1])]),
    (re.compile(r"^GPU models and configuration\s+:\s*(?:(?:GPU|NIC) \d+: )?(.+)$"),
     lambda m: _env("gpu.model", m[1])),
    # Multi-node boxes print the aggregate empty and then one `GPU 0: NVIDIA GB200`
    # per card; the post-pass below promotes that to gpu.model only when nothing
    # else stated it.
    (re.compile(r"^GPU \d+: (.+)$"), lambda m: _env("gpu.card", m[1])),
    (re.compile(r"^\[pip3\] (torch|triton|transformers|numpy|flashinfer-python)==(\S+)\b"),
     lambda m: [(f"env.pip.{m[1]}", m[2])]),
    # collect_env ends with the environment the engine ran in. Any NAME=value there is a
    # setting, so the list is general — minus device selection and per-host cache paths,
    # which differ between two boxes started with the same flags and must not read as a
    # difference.
    (re.compile(r"^(?!(?:CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|VLLM_CACHE_ROOT)=)"
                r"([A-Z][A-Z0-9_]{2,})=(.+)$"),
     lambda m: [(f"env.var.{m[1]}", m[2].strip())]),
    # SGLang prints its own environment dump, `name: value` one per line, half of which is
    # package inventory. The versions below decide which kernels load.
    (re.compile(r"^Python: (\S+)"), lambda m: _env("env.python", m[1])),
    (re.compile(r"^PyTorch: (\S+)"), lambda m: _env("env.torch", m[1])),
    (re.compile(r"^sglang: (\S+)"), lambda m: _env("engine.version", m[1])),
    (re.compile(r"^sglang-kernel: (\S+)"), lambda m: _env("env.pip.sglang-kernel", m[1])),
    (re.compile(r"^flashinfer_python: (\S+)"), lambda m: _env("env.pip.flashinfer-python", m[1])),
    (re.compile(r"^triton: (\S+)"), lambda m: _env("env.pip.triton", m[1])),
    (re.compile(r"^transformers: (\S+)"), lambda m: _env("env.pip.transformers", m[1])),
    (re.compile(r"^numpy: (\S+)"), lambda m: _env("env.pip.numpy", m[1])),
    (re.compile(r"^CUDA_HOME: (\S+)"), lambda m: _env("env.var.CUDA_HOME", m[1])),
    (re.compile(r"^NVCC: (.+)$"), lambda m: _env("env.nvcc", m[1])),
    (re.compile(r"^CUDA Driver Version: (\S+)"), lambda m: _env("env.cuda_driver", m[1])),
    # SGLang runtime: the radix cache it built, the transfer backend it chose, and a
    # scheduler that died before serving — none of it appears in `server_args`.
    (re.compile(r"^Tree cache initialized: (.+)$"), lambda m: [("cache.radix", m[1].strip())]),
    (re.compile(r"^NIXL KVManager initialized with backend: (\S+)"),
     lambda m: [("transfer.nixl_backend", m[1])]),
    (re.compile(r"^Scheduler hit an exception"), lambda m: [("startup", "failed (scheduler)")]),
    (re.compile(r"^ulimit (soft|hard): (\d+)"), lambda m: [(f"env.ulimit.{m[1]}", int(m[2]))]),
    # Model loading cost: memory footprint and wall time. Both V1 (gpu_model_runner)
    # and V2 (model_runner) print the same message, and 0.23 printed it without the
    # word `memory` — a rule pinned to the newer wording silently lost the older log.
    (re.compile(r"Model loading took ([\d.]+) GiB(?: memory)? and ([\d.]+) seconds"),
     lambda m: [("model_loading.gib", float(m[1])), ("model_loading.seconds", float(m[2]))]),
    # The loading phase splits into three timed steps. Comparing "startup is slower"
    # between two versions is meaningless until you know which step grew: constructing
    # the module, reading shards, or compiling. The scratch line says whether the run
    # even had a disk cache to read from.
    (re.compile(r"Loading model from scratch"), lambda m: [("weights.cache", "miss (from scratch)")]),
    (re.compile(r"Initializing model took ([\d.]+) seconds"),
     lambda m: [("model_init.seconds", float(m[1]))]),
    (re.compile(r"Loading weights took ([\d.]+) seconds"),
     lambda m: [("weights.load_s", float(m[1]))]),
    # torch.compile outcome: artifact count/size reflects compile behaviour; total
    # wall time is comparable between versions on the same machine.
    (re.compile(r"collected artifacts: (.+?) total"),
     lambda m: [("compile.artifacts", m[1].strip())]),
    (re.compile(r"torch.compile took ([\d.]+) s in total"),
     lambda m: [("compile.total_s", float(m[1]))]),
    (re.compile(r"Pinned memory is not available.*falling back to (.+?) for UVA buffers"),
     lambda m: [("uva_buffers", f"fallback: {m[1]}")]),
    # Memory headroom at startup. The gap between free and budgeted memory is what a
    # later OOM is judged against, and utilization can be "set" yet not reached.
    (re.compile(r"Free memory on device \(([\d.]+)/([\d.]+) GiB\) on startup\. Desired GPU memory utilization is \(([\d.]+), ([\d.]+) GiB\)"),
     lambda m: [("gpu.free_gib", float(m[1])),
                ("gpu.total_gib", float(m[2])),
                ("gpu.utilization", float(m[3])),
                ("gpu.budget_gib", float(m[4]))]),
    # CUDA graph memory profiling is on by default since v0.21.0, and it changes what a
    # given --gpu-memory-utilization actually buys: the line itself prints the equivalent
    # value without profiling and the value needed to keep the old KV cache size. Comparing
    # two versions that differ only in this default is exactly the silent change servediff
    # exists to surface, so both numbers are kept rather than the flag.
    (re.compile(r"CUDA graph memory profiling is (enabled|disabled) \(default since v([\d.]+)\)"),
     lambda m: [("cuda_graph.profiling", f"{m[1]} (default since v{m[2]})")]),
    (re.compile(r"equivalent to --gpu-memory-utilization=([\d.]+) without CUDA graph memory profiling"),
     lambda m: [("gpu.utilization_effective", float(m[1]))]),
    (re.compile(r"increase --gpu-memory-utilization to ([\d.]+)\."),
     lambda m: [("gpu.utilization_to_match_kv_cache", float(m[1]))]),
    # Where the weights came from and whether the loader could prefetch them. On a slow
    # filesystem this is the difference between a 10 s and a 100 s startup, and the
    # checkpoint size line is the only place the loader states what it measured.
    (re.compile(r"Filesystem type for checkpoints: ([\w.]+)\. Checkpoint size: ([\d.]+) GiB\. Available RAM: ([\d.]+) GiB"),
     lambda m: [("weights.filesystem", m[1]),
                ("weights.checkpoint_gib", float(m[2])),
                ("weights.available_ram_gib", float(m[3]))]),
    (re.compile(r"Auto-prefetch is (disabled) because the filesystem \(([\w.]+)\) is not a recognized network FS"),
     lambda m: [("weights.prefetch", f"{m[1]} ({m[2]} is not a network FS)")]),
    # What the server will actually serve, and whether it is allowed to fan out.
    (re.compile(r"Supported tasks: (.+)"), lambda m: [("server.tasks", m[1].strip())]),
    (re.compile(r"Scale-out endpoints are (disabled)"), lambda m: [("server.scale_out", m[1])]),
    (re.compile(r"chat template content format to be '([^']+)'"),
     lambda m: [("serving.chat_template_content_format", m[1])]),
    (re.compile(r"Skipping FlashInfer autotune because it is (disabled)"),
     lambda m: [("kernel.flashinfer_autotune", m[1])]),
    # A silent compile-time gate: without enough SMs the fast GEMM autotune path is
    # skipped, which changes kernel selection without changing any argument.
    (re.compile(r"Not enough SMs to use max_autotune_gemm mode"),
     lambda m: [("compile.autotune_gemm", "disabled (not enough SMs)")]),
    # Ops left native vs dispatched to a fused kernel, as the platform finally decided.
    (re.compile(r"Final IR op priority after setting platform defaults: (.+)$"),
     lambda m: _ir_op_facts(m[1])),
    # Kernel JIT work inside the startup window, and whether late compiles are silent.
    (re.compile(r"JIT kernel warmup finished in ([\d.]+)s"),
     lambda m: [("jit.warmup_s", float(m[1]))]),
    (re.compile(r"monitored JIT compilations during inference will use mode=(\w+)"),
     lambda m: [("jit.monitor_mode", m[1])]),
    # torch.compile on the backbone, split into the costs it actually spends time on: the
    # bytecode transform, one graph per compile range, and a single profiling pass. An
    # upgrade can add seconds here without changing a single flag, and one total would
    # hide which of the three moved.
    (re.compile(r"Dynamo bytecode transform time: ([\d.]+) s"),
     lambda m: [("compile.dynamo_s", float(m[1]))]),
    (re.compile(r"Compiling a graph for compile range \((\d+), (\d+)\) takes ([\d.]+) s"),
     lambda m: [("compile.range", f"{m[1]},{m[2]}"), ("compile.graph_s", float(m[3]))]),
    (re.compile(r"Initial profiling/warmup run took ([\d.]+) s"),
     lambda m: [("compile.warmup_s", float(m[1]))]),
    # Serving torch threads are cut to one by default; the pair matters, because "1" alone
    # cannot say whether it was cut from 16 (a CPU-heavy box) or from 4.
    (re.compile(r"Reducing Torch threads from (\d+) to (\d+) for serving"),
     lambda m: [("cpu.torch_threads", f"{m[1]} -> {m[2]}")]),
    # The same cut applied to intra-op parallelism, and the multiprocessing start method
    # the workers were forced onto: both change timing without changing a flag.
    (re.compile(r"Reducing Torch parallelism from (\d+) threads to (\d+)"),
     lambda m: [("cpu.torch_parallelism", f"{m[1]} -> {m[2]}")]),
    (re.compile(r"We must use the `spawn` multiprocessing start method"),
     lambda m: [("worker.mp_method", "spawn")]),
    # Which attention backend the platform actually picked. A flag asks for one; this
    # line says what was chosen after the check, and FlashInfer vs FlashAttention is a
    # different kernel set for the same command line.
    (re.compile(r"Using (AttentionBackendEnum\.\w+) backend"), lambda m: [("attn.backend", m[1].split(".")[-1])]),
    (re.compile(r"Using (Eagle\d*) auxiliary layers from config: (.+)$"),
     lambda m: [("spec_decode.eagle_layers", f"{m[1]}: {m[2].strip()}")]),
    # Batching and scheduling decisions that decide the shape of every forward pass.
    (re.compile(r"Chunked prefill is (enabled|disabled)(?: with max_num_batched_tokens=(\d+))?"),
     lambda m: [("scheduler.chunked_prefill",
                 m[1] if not m[2] else f"{m[1]} (max_num_batched_tokens={m[2]})")]),
    (re.compile(r"Asynchronous scheduling is (enabled|disabled)"), lambda m: [("scheduler.async", m[1])]),
    # A compiled graph reused across compile ranges is a real startup saving, and its
    # range is what a later recompile is compared against.
    (re.compile(r"Cache the graph of compile range \((\d+), (\d+)\) for later use"),
     lambda m: [("compile.cached_range", f"{m[1]},{m[2]}")]),
    # Kernel-level failures that do not stop startup but do change what runs: a wrong
    # kernel here is degraded output or a slower path, not a crash, so it is easy to miss.
    (re.compile(r"CUDA error \((.+?)\): (.+)$"), lambda m: [("kernel.cuda_error", f"{m[2].strip()} ({m[1].strip()})")]),
    (re.compile(r"vLLM is using nccl==([\d.]+)"), lambda m: [("nccl.version", m[1])]),
    (re.compile(r"Using FlashInfer for top-p & top-k sampling"),
     lambda m: [("sampling.topk_topp_backend", "flashinfer")]),
    (re.compile(r"FlashInfer resolved query dtypes: (.+)$"),
     lambda m: [("kernel.flashinfer_dtypes", m[1].strip().rstrip(","))]),
    # Data-parallel topology, as the engine laid it out rather than as requested.
    (re.compile(r"DP group leader: node_rank=(\d+), node_rank_within_dp=(\d+)"),
     lambda m: [("parallel.dp_leader", f"node {m[1]} / dp {m[2]}")]),
    # A worker dying is a startup failure even when the API server keeps printing.
    (re.compile(r"Worker proc .+? died unexpectedly"),
     lambda m: [("startup", "failed (worker died)")]),
    # Sampling defaults are silently taken from the model's generation_config.json;
    # two runs of the same flags can sample differently because of it.
    (re.compile(r"generation_config\.json[`'\":]+ ?(.+)$"),
     lambda m: [("sampling.overridden", m[1].strip().strip("`"))]),
    # A silent numeric change: the checkpoint's dtype was coerced to something the GPU
    # path wants. Nothing in the flags says so, and it changes every result.
    (re.compile(r"Casting (torch\.\w+) to (torch\.\w+)"),
     lambda m: [("model.cast", f"{m[1]} -> {m[2]}")]),
    # An optional kernel library that is installed but not importable: the build silently
    # falls back to a slower or less capable path, and nothing in the flags records it.
    (re.compile(r"Module (\S+) was found but failed to import"),
     lambda m: [("kernel.module_missing", m[1])]),
    (re.compile(r"ImportError: (\S+): cannot open shared object file"),
     lambda m: [("kernel.missing_lib", m[1])]),
    # Startup outcome. Without it, facts missing after a crash look like "not printed".
    (re.compile(r"Application startup complete"), lambda m: [("startup", "ready")]),
    (re.compile(r"EngineCore failed to start|Engine core initialization failed"),
     lambda m: [("startup", "failed")]),
    (re.compile(r"AsyncLLM output_handler failed"),
     lambda m: [("startup", "failed (output handler)")]),
    # Exception classes arrive fully qualified in some versions
    # (`vllm.v1.engine.exceptions.EngineDeadError: …`), so the type may be mid-line.
    (re.compile(r"((?:RuntimeError|ValueError|IndexError|torch\.OutOfMemoryError|AssertionError|AttributeError|KeyError|TypeError|EngineDeadError): .+)$"),
     lambda m: [("startup.error", m[1].strip())]),
    # SGLang.
    (re.compile(r"Disable prefill CUDA graph because"), lambda m: [("cuda_graph.prefill", "disabled")]),
    (re.compile(r"The server is fired up and ready to roll"), lambda m: [("startup", "ready")]),
    (re.compile(r"KV Cache is allocated\. dtype: (\S+?),"), lambda m: [("kv_cache.dtype", m[1])]),
    # SGLang prints the exact launch command, including the flags it defaulted.
    (re.compile(r"sgl start command: (.+)$"), lambda m: [("server.command", m[1].strip())]),
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


def _ir_op_facts(text: str) -> list[tuple[str, str]]:
    """One fact per op, not one long blob for the whole config.

    The full IrOpPriorityConfig line is far wider than the table, so two different
    configs truncate to the same visible string and the row says "changed" while
    looking identical. Splitting by op puts the difference on its own short key.
    """
    facts = []
    for op, candidates in re.findall(r"(\w+)=\[([^\]]*)\]", text):
        chosen = "|".join(c.strip().strip("'\"") for c in candidates.split(",") if c.strip())
        facts.append((f"kernel.ir_op.{op}", chosen))
    return facts or []


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
        line = BRACKET_TS.sub("", ANSI.sub("", raw))
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

    card = facts.pop("gpu.card", None)
    if card is not None and "gpu.model" not in facts:
        facts["gpu.model"] = card
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


def series(parsed_list: list[dict], raw_args: bool = False) -> list[dict]:
    """Compare N logs at once: one row per fact, one value per log.

    Regression threads usually carry three or more runs (0.29 / 0.31 / 0.32), and
    two pairwise diffs lose the middle column. Rows that hold the same value across
    every log are marked `same`.
    """
    keys = set()
    for parsed in parsed_list:
        keys |= set(parsed["facts"])
    rows = []
    for key in sorted(keys):
        values = [_shown(parsed["facts"].get(key)) for parsed in parsed_list]
        rows.append({"key": key, "values": values,
                     "status": "same" if all(v == values[0] for v in values) else "changed"})
    if raw_args and len({parsed["engine"] for parsed in parsed_list}) == 1:
        keys = set()
        for parsed in parsed_list:
            keys |= set(parsed["args"])
        for key in sorted(keys):
            values = [parsed["args"].get(key, "<absent>") for parsed in parsed_list]
            if any(v != values[0] for v in values):
                rows.append({"key": f"raw_arg.{key}", "values": values, "status": "changed"})
    return rows


def _version_tuple(facts) -> tuple | None:
    fact = facts.get("engine.version")
    if not fact:
        return None
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", str(fact.values[0]))
    return tuple(int(part) for part in match.groups()) if match else None


def _is_single_rank(fact) -> bool:
    value = fact.value
    values = value if isinstance(value, list) else [value]
    return all(v == 1 for v in values)


def absence_notes(parsed: dict) -> list[str]:
    """Name the reason a log cannot yield a fact, where the tool can actually tell.

    A fact absent from *both* logs leaves no diff row at all, so the summary says
    "identical" about something that was never observed. For the all-reduce path
    that silence carries the most weight: a single-rank run has no group to choose
    a backend for, and since v0.29.0 the FlashInfer auto-selection line is logged
    at DEBUG, so a default INFO log never contains it. Leaving the reader to guess
    is how "absent" gets read as "off".
    """
    facts = parsed["facts"]
    if "backend.allreduce_dispatch" not in facts:
        world = facts.get("parallel.world_size")
        if world is None:
            return ["no all-reduce facts and no rank line: cannot tell whether this run had more than "
                    "one rank, so an absent all-reduce path here is inconclusive"]
        if _is_single_rank(world):
            return ["no all-reduce facts: world_size=1, so there is no all-reduce group and no backend "
                    "is chosen; a TP>1 run is what makes this path observable"]
        return [f"no all-reduce facts although world_size={world.value}: the selection is printed once per "
                "group at engine init, so a log that starts after initialisation cannot show it"]
    version = _version_tuple(facts)
    if "backend.allreduce_flashinfer" not in facts and version and version >= (0, 29, 0):
        return ["backend.allreduce_flashinfer is absent although the dispatch chain printed: since "
                f"v{'.'.join(map(str, version))} that line is logged at DEBUG, so a default INFO log never "
                "contains it; the dispatch chain is the signal, not this fact"]
    return []


def _fmt(value) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= 60 else text[:57] + "..."


def _md_cell(value) -> str:
    """Render a diff cell for a markdown table: escape pipes, collapse lists."""
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text.replace("|", "\\|").replace("\n", " ")


def _render_markdown(*, path_a: str, path_b: str, engine_a, engine_b,
                     rows: list[dict], warnings: list[str], notes: list[str] = [],
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
    for n in notes:
        lines.append(f"> ℹ️ {n}")
    if notes:
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


def _short(text: str, width: int, keep_tail: int = 0) -> str:
    """Trim to `width`. With `keep_tail`, elide the middle so the ends stay visible: in a
    multi-column table two values that differ only at the end must not render the same."""
    if len(text) <= width:
        return text
    if not keep_tail or width - 3 - keep_tail < 4:
        return text[: width - 3] + "..."
    return text[: width - 3 - keep_tail] + "..." + text[-keep_tail:]


def _render_text_series(labels: list[str], rows: list[dict], width: int = 30) -> str:
    lines = [f"{'fact':32s} " + " ".join(f"{_short(label, width):{width}s}" for label in labels)]
    for row in rows:
        cells = " ".join(f"{_short(_fmt(value), width, keep_tail=8):{width}s}" for value in row["values"])
        lines.append(f"{row['key']:32s} {cells}".rstrip())
    return "\n".join(lines)


def _render_markdown_series(*, labels: list[str], rows: list[dict], differing: int, identical: int) -> str:
    head = "| fact | " + " | ".join(_md_cell(label) for label in labels) + " |"
    rule = "|---|" + "---|" * len(labels)
    body = ["| `" + row["key"] + "` | "
            + " | ".join("`" + _md_cell(_fmt(value)) + "`" for value in row["values"]) + " |"
            for row in rows]
    return "\n".join([head, rule, *body, "",
                      f"{differing} fact(s) differ across {len(labels)} logs; {identical} identical. "
                      "`<absent>` means the log did not print it, not that it was off."])


def _version() -> str:
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("servediff")
    except Exception:
        return "unknown"


def _label(path) -> str:
    return "stdin" if str(path) == "-" else str(path)


class InputError(Exception):
    """An input path could not be read. Callers map it to exit 2, never to 1."""


def _read_log(path) -> str:
    """Read a log path, or stdin when the path is `-`.

    Logs usually live on the box that runs the server, so the useful shape is
    `ssh node 'cat serve.log' | servediff show -` rather than scp first.
    """
    if str(path) == "-":
        return sys.stdin.buffer.read().decode("utf-8", errors="replace")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        # Exit 1 means "the two sides differ" to every caller and CI script, so an
        # unreadable path must not borrow it: raise a traceback here and a typo looks
        # like a regression.
        raise InputError(f"{path}: {exc.strerror or exc}") from exc


def _audit_gate(max_shapes: int | None, shapes: int) -> int:
    """Audit's exit code with a threshold: 1 means 'the parser is behind this log', not
    'the log is different', so it does not collide with diff's 1."""
    if max_shapes is None:
        return 0
    if shapes > max_shapes:
        print(f"audit: {shapes} shape(s) worth a look is above --max-shapes {max_shapes}",
              file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"servediff {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", help="effective config facts in one log")
    show.add_argument("log", type=Path, help="server log, or - to read stdin")
    show.add_argument("--json", action="store_true")
    d = sub.add_parser("diff", help="diff two server logs, or two `servediff env` snapshots")
    d.add_argument("baseline", type=Path, help="server log or env snapshot; - reads stdin")
    d.add_argument("candidate", type=Path, help="server log or env snapshot; - reads stdin")
    d.add_argument("--raw-args", action="store_true", help="also diff every raw argument (same engine only)")
    d.add_argument("--json", action="store_true")
    d.add_argument("--markdown", action="store_true", help="render a GitHub-flavored report (paste into issues)")
    d.add_argument("--all", action="store_true", help="also print unchanged facts")
    d.add_argument("--skip-absent", action="store_true",
                   help="hide facts one side never printed, so a failed run's tail does not"
                        " bury the rows that both logs can be compared on")
    s = sub.add_parser("series", help="compare three or more server logs in one table")
    s.add_argument("logs", type=Path, nargs="+", help="server logs, in the column order you want; - reads stdin")
    s.add_argument("--raw-args", action="store_true", help="also compare every raw argument (same engine only)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--markdown", action="store_true", help="render a GitHub-flavored report (paste into issues)")
    s.add_argument("--all", action="store_true", help="also print facts identical across every log")
    e = sub.add_parser("env", help="snapshot package versions, GPU and engine env vars of a server environment")
    e.add_argument("--python", help="interpreter of the server's environment (default: this one)")
    e.add_argument("-o", "--output", type=Path, help="write JSON here instead of stdout")
    c = sub.add_parser("capture", help="run a server command, tee its log, and snapshot its environment")
    c.add_argument("-o", "--output", type=Path, default=Path("servediff-capture.log"),
                   help="log file (default: servediff-capture.log)")
    c.add_argument("--python", help="interpreter for the env snapshot (default: this one)")
    c.add_argument("run_cmd", nargs=argparse.REMAINDER, metavar="-- command",
                   help="server command, after --")
    sub.add_parser("demo", help="diff two logs shipped with the package; needs no log files")
    a = sub.add_parser("audit", help="list log lines servediff reads nothing from (report a missed fact)")
    a.add_argument("logs", type=Path, nargs="+", help="server logs, or - to read stdin")
    a.add_argument("--top", type=int, default=0, help="only the N most common shapes worth a look")
    a.add_argument("--everything", action="store_true", help="also list the noise shapes")
    a.add_argument("--markdown", action="store_true",
                   help="wrap the report in a code fence, ready to paste into an issue")
    a.add_argument("--max-shapes", type=int, default=None, metavar="N",
                   help="exit 1 when the number of shapes worth a look is above N, so a"
                        " pipeline can fail when a log format grows past the parser")
    args = parser.parse_args(argv)

    if args.command == "audit":
        if sum(1 for path in args.logs if str(path) == "-") > 1:
            print("audit: stdin can be read once, so at most one entry may be '-'", file=sys.stderr)
            return 2
        from servediff import audit as auditmod

        blocks = []
        # Counted separately from the printed report so a threshold decision never depends
        # on parsing the report text back.
        shapes_worth_a_look = 0
        for path in args.logs:
            try:
                text = _read_log(path)
            except InputError as exc:
                print(f"audit: {exc}", file=sys.stderr)
                return 2
            counts, samples = auditmod.unread_samples(text)
            shapes_worth_a_look += len(auditmod.split(counts, samples)[0])
            blocks.append(auditmod.report(Path(_label(path)), text,
                                          limit=args.top, everything=args.everything))
        if args.markdown:
            # The issue form asks for the engine, the line, and this report, so step three
            # should be one paste. The fence is made longer than any run of backticks in
            # the text: log messages do quote flags with them, and a stray ``` would end
            # the block early and truncate exactly what the reader needs.
            body = "\n\n".join(blocks)
            longest = max((len(run) for run in re.findall("`+", body)), default=0)
            fence = "`" * max(3, longest + 1)
            print(f"{fence}text")
            print(body)
            print(fence)
            return _audit_gate(args.max_shapes, shapes_worth_a_look)
        print("\n\n".join(blocks))
        if args.max_shapes is not None:
            print(f"\naudit: {shapes_worth_a_look} shape(s) worth a look, threshold"
                  f" {args.max_shapes}", file=sys.stderr)
        return _audit_gate(args.max_shapes, shapes_worth_a_look)

    if args.command == "demo":
        # A fresh install has nothing to diff, and the tool is useless until you find two
        # logs. These two excerpts are public lines from vllm#47047, shipped inside the
        # package, so the first command a new user runs produces real output.
        from importlib.resources import files
        import tempfile

        print("servediff demo: two excerpt logs from vllm#47047 (v0.23, same flags, container"
              " MoE backend differs)\n", file=sys.stderr)
        base = files("servediff").joinpath("examples")
        with tempfile.TemporaryDirectory() as tmp:
            names = ["vllm-47047-v0.23-default.log", "vllm-47047-v0.23-marlin.log"]
            paths = []
            for name in names:
                dest = Path(tmp) / name
                dest.write_text(base.joinpath(name).read_text(encoding="utf-8"), encoding="utf-8")
                paths.append(dest)
            code = main(["diff", *map(str, paths)])
        print(f"\ndemo: exit {code}. The two bundled logs do differ (the MoE backend, plus a KV"
              " cache line only one of them prints), so `diff` reports 1 here — 1 means"
              " \"the inputs differ\", not \"the tool failed\"; 2 would mean it could not"
              " read them.", file=sys.stderr)
        return code

    if args.command == "capture":
        cmd = args.run_cmd
        if cmd[:1] == ["--"]:
            cmd = cmd[1:]
        if not cmd:
            print("capture: provide a command after --, e.g. servediff capture -- vllm serve <model>",
                  file=sys.stderr)
            return 2
        # Snapshot the environment before the server starts, so we capture what it launched with.
        # When the command itself names a Python launcher (`capture -- python -m vllm...`), that
        # launcher *is* the server's environment; snapshotting our own would describe the wrong
        # venv and make a later snapshot diff compare two copies of servediff.
        snapshot_python = args.python
        if snapshot_python is None and cmd:
            launcher = Path(cmd[0]).name.lower()
            if re.fullmatch(r"python(\d+(\.\d+)*)?((\.exe)?)", launcher):
                snapshot_python = cmd[0]
        try:
            snap = envsnap.snapshot(snapshot_python)
        except (RuntimeError, OSError) as exc:
            print(f"servediff capture: env snapshot failed: {exc}", file=sys.stderr)
            return 2
        snap_path = args.output.with_suffix(args.output.suffix + ".env.json")
        snap_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                             encoding="utf-8")
        print(f"env snapshot -> {snap_path}  (interpreter: {snap.get('executable') or snapshot_python})",
              file=sys.stderr)
        if snapshot_python is None:
            print("  note: the command does not name a python launcher, so this snapshot describes"
                  " the interpreter servediff runs on; pass --python <the server's python> if the"
                  " server lives in another venv", file=sys.stderr)
        print(f"log -> {args.output}  (running: {' '.join(shlex.quote(p) for p in cmd)})", file=sys.stderr)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
        except (FileNotFoundError, OSError) as exc:
            print(f"capture: cannot start {cmd[0]}: {exc}", file=sys.stderr)
            return 2
        with open(args.output, "w", encoding="utf-8", errors="replace") as logf:
            interrupted = False
            try:
                for line in proc.stdout:
                    logf.write(line)
                    logf.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                # Ctrl-C is how a server run ends in practice. Without this it bubbled
                # out as a traceback with exit 1, which reads as "the inputs differ".
                interrupted = True
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        if not interrupted:
            proc.wait()
        else:
            print(f"\ninterrupted; server stopped, log kept: {args.output}"
                  f" (startup lines printed before the interruption are still in it)", file=sys.stderr)
            return 130
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

    if args.command == "series":
        if len(args.logs) < 3:
            print("series: give at least three logs; for two use `servediff diff`", file=sys.stderr)
            return 2
        if sum(1 for path in args.logs if str(path) == "-") > 1:
            print("series: stdin can be read once, so at most one entry may be '-'", file=sys.stderr)
            return 2
        try:
            parsed_list = [parse(_read_log(path)) for path in args.logs]
        except InputError as exc:
            print(f"series: {exc}", file=sys.stderr)
            return 2
        for path, parsed in zip(args.logs, parsed_list):
            if not parsed["facts"]:
                print(f"{_label(path)}: no vLLM or SGLang configuration lines recognised", file=sys.stderr)
                return 2
        labels = ["stdin" if str(name) == "-" else Path(name).name for name in args.logs]
        rows = series(parsed_list, raw_args=args.raw_args)
        differing = [row for row in rows if row["status"] == "changed"]
        shown = rows if args.all else differing
        if args.json:
            print(json.dumps({"logs": labels, "differences": len(differing), "rows": shown},
                             indent=2, ensure_ascii=False))
        elif args.markdown:
            print(_render_markdown_series(labels=labels, rows=shown, differing=len(differing),
                                          identical=len(rows) - len(differing)))
        else:
            print(_render_text_series(labels, shown))
            print(f"\n{len(differing)} fact(s) differ across {len(labels)} logs;"
                  f" {len(rows) - len(differing)} identical. <absent> means the log did not print it,"
                  " not that it was off.")
        return 1 if differing else 0

    if args.command == "show":
        try:
            parsed = parse(_read_log(args.log))
        except InputError as exc:
            print(f"show: {exc}", file=sys.stderr)
            return 2
        if not parsed["facts"]:
            print(f"{_label(args.log)}: no vLLM or SGLang configuration lines recognised", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps({k: {"value": f.value, "evidence": f.evidence} for k, f in parsed["facts"].items()},
                             indent=2, ensure_ascii=False))
        else:
            for key, fact in sorted(parsed["facts"].items()):
                where = fact.evidence[0][0]
                print(f"{key:32s} {_fmt(fact.value):60s} line {where if where else '-'}")
        return 0

    if str(args.baseline) == "-" and str(args.candidate) == "-":
        print("diff: stdin can be read once, so only one side may be '-'", file=sys.stderr)
        return 2
    try:
        text_a = _read_log(args.baseline)
        text_b = _read_log(args.candidate)
    except InputError as exc:
        print(f"diff: {exc}", file=sys.stderr)
        return 2
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
                path_a=_label(args.baseline), path_b=_label(args.candidate),
                engine_a=None, engine_b=None,
                rows=shown, warnings=[],
                differing=len(key), identical=len(rows) - len(key),
                all_rows=bool(args.all)))
        else:
            shown = rows if args.all else key
            print(f"{'fact':32s} {'baseline':40s} {'candidate':40s} status")
            for r in shown:
                print(f"{r['key']:32s} {_short(_fmt(r['baseline']), 40):40s}"
                      f" {_short(_fmt(r['candidate']), 40):40s} {r['status']}")
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
    all_differing = [r for r in rows if r["status"] != "same"]
    # A run that died at startup leaves every later fact <absent>, so one-sided rows
    # outnumber real differences roughly 10:1 and bury them. --skip-absent keeps only
    # the facts both logs printed: what changed on the same code path.
    differing = all_differing
    hidden_absent: list[dict] = []
    if args.skip_absent:
        hidden_absent = [r for r in all_differing if r["status"] != "changed"]
        differing = [r for r in all_differing if r["status"] == "changed"]
    identical = len(rows) - len(all_differing)
    notes = []
    for side, parsed in (("baseline", a), ("candidate", b)):
        notes.extend(f"{side}: {note}" for note in absence_notes(parsed))
    if args.skip_absent and all_differing:
        notes.append(f"{len(hidden_absent)} difference(s) hidden by --skip-absent: one side"
                     " never printed those lines, so they can only be listed, not compared")
    if args.json:
        print(json.dumps({"differences": len(all_differing),
                          "compared_both_sides": len(differing),
                          "hidden_absent": len(hidden_absent),
                          "notes": notes, "rows": differing if args.skip_absent else rows},
                         indent=2, ensure_ascii=False, default=str))
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
            path_a=_label(args.baseline), path_b=_label(args.candidate),
            engine_a=a["engine"], engine_b=b["engine"],
            rows=shown_rows, warnings=warnings, notes=notes,
            differing=len(differing), identical=identical,
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
        for note in notes:
            print(f"note: {note}")
        if notes:
            print()
        print(f"{'fact':32s} {'baseline':40s} {'candidate':40s} status")
        for r in rows if args.all else differing:
            # Mark truncation instead of cutting a value mid-word: a cell that ends
            # in `TP ran` reads like the value itself, not like a clipped column.
            print(f"{r['key']:32s} {_short(_fmt(r['baseline']), 40):40s}"
                  f" {_short(_fmt(r['candidate']), 40):40s} {r['status']}")
        line = f"\n{len(differing)} difference(s); {identical} fact(s) identical."
        if hidden_absent:
            line += (f" {len(hidden_absent)} difference(s) skipped because one side never printed"
                     " those lines (--skip-absent).")
        print(line + " Absent means the log did not print it, not that it was off.")
    return 1 if all_differing else 0


if __name__ == "__main__":
    sys.exit(main())
