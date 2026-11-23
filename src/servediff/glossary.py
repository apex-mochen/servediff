"""What each fact key means, and what it does not tell you.

`show` prints our reading of a log. A reader who does not know what
`kv_cache.max_concurrency` counts cannot use that reading, and the only place the
meaning lived was the comment above the regex that produced it -- in the source.

One entry per key, or per family when the family shares a meaning (`env.pip.*` is
always "this wheel was importable"). `lookup` takes the longest matching prefix, so
an exact key wins over its family.

Every entry keeps the same shape: what the number is, what it is measured in, and
the thing a reader is most likely to assume wrongly about it. The `trap` field is the
one that pays for the entry: a fact that reads as an answer but answers a slightly
different question is how a diff gets misread.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gloss:
    what: str
    unit: str
    trap: str = ""


GLOSSARY: dict[str, Gloss] = {
    # -- environment ----------------------------------------------------------------
    "env": Gloss(
        "The environment the server ran in: the collect_env block vLLM asks every bug "
        "report to paste, plus SGLang's equivalent. `env.pip.*` says whether a wheel was "
        "importable and at which version; `env.var.*` is an environment variable the "
        "server printed; `env.ulimit.*` is the process limit.",
        "versions, paths, environment variables",
        "a version here is a wheel that imported, not necessarily the kernel that ran, "
        "and two runs on the same versions can still differ in what they selected -- "
        "read env next to the other facts, never instead of them",
    ),
    # -- the run itself -------------------------------------------------------------
    "engine": Gloss(
        "Which build served: version string, git sha when the log carries one, platform "
        "it ran on, engine.init_s for construction time. Two logs whose facts are identical "
        "but whose versions differ are not the same run, because the reading rules change.",
        "version, sha, seconds",
        "init_s is engine construction (weights, compile, cache sizing) and excludes a "
        "model download that happened before the process started",
    ),
    "engine.init_s": Gloss(
        "How long the engine took to construct, as the engine itself timed it.",
        "seconds",
        "it ends when init returns, so the first CUDA graph capture and any lazy warmup "
        "that happens on the first request are outside it",
    ),
    "model_runner": Gloss(
        "A marker, not a number: the log came from the V1 engine (gpu_model_runner).",
        "a label",
        "the V0 engine printed a different runner line, so a V0-vs-V1 comparison shows as "
        "this key appearing rather than changing",
    ),
    "model": Gloss(
        "What was served: architecture, max length, supported tasks, and the dtype the "
        "weights were cast to. model.cast is the pair `checkpoint dtype -> running dtype`.",
        "names, token counts, dtypes",
        "architecture is what the config declared; the backend keys are what the build "
        "actually ran with for that architecture",
    ),
    "model_loading": Gloss(
        "The loader's own accounting of reading the checkpoint: GiB resident after load "
        "and the seconds it took.",
        "GiB, seconds",
        "the GiB here is what the weights occupy, not what the card used -- CUDA context, "
        "allocator and non-torch buffers are outside it",
    ),
    "model_init": Gloss(
        "Time spent constructing the nn.Module before weights are read.",
        "seconds",
        "on a big model this is often small next to model_loading; a change here points at "
        "transformers or model code, not at the disk",
    ),
    "weights": Gloss(
        "Where the weights came from and how they were read: format, filesystem, cache, "
        "download and load seconds, checkpoint size, dtype, available RAM, and whether "
        "prefetch was possible (weights.prefetch says why not).",
        "GiB, seconds, MiB/s",
        "download_s is the network leg only; a slow startup on a local disk shows in "
        "load_s and in `weights.prefetch: disabled (...)`, not in download_s",
    ),
    # -- memory -------------------------------------------------------------------
    "gpu": Gloss(
        "Per-device memory accounting, phase by phase: total and free before weights, the "
        "budget the utilization fraction allows, what was consumed after profiling, what "
        "CUDA graphs reserved, and the peak activation the profile run saw.",
        "GiB; utilization is a fraction of total",
        "free_gib is measured before anything is loaded, and the OOM that follows is "
        "judged against consumed_gib -- the two are not the same pool",
    ),
    "gpu.model": Gloss(
        "The card the log named.",
        "a name",
        "vLLM prints it once per process in some versions; on a multi-GPU box it is the "
        "card the rank reporting happened to run on",
    ),
    "gpu.utilization_effective": Gloss(
        "The utilization fraction the run actually reached (consumed / total), which vLLM "
        "prints as `equivalent to --gpu-memory-utilization=X`.",
        "fraction of total device memory",
        "the value you passed and the value reached are different facts; a run that asked "
        "for 0.90 and reached 0.77 has 13% of the card idle, and that is the number to "
        "compare before blaming the model",
    ),
    "gpu.utilization_to_match_kv_cache": Gloss(
        "What --gpu-memory-utilization you would have had to pass to get the KV cache you "
        "asked for.",
        "fraction of total device memory",
        "it is advice printed after the engine failed to fit the cache, so it describes a "
        "configuration that was never run",
    ),
    "kv_cache": Gloss(
        "The cache the engine sized after profiling: GiB and tokens it holds, the sequence "
        "length and concurrency that implies, its dtype and block layout, and anything the "
        "engine had to pad or set aside.",
        "GiB for sizes, tokens for tokens, sequences for concurrency",
        "this is what the engine decided the cache is, after auto-fitting; what you asked "
        "for is in arg.kv_cache_dtype / kv_cache.requested_gib",
    ),
    "kv_cache.max_concurrency": Gloss(
        "How many concurrent sequences the cache holds at the per-request token count the "
        "same sentence names.",
        "sequences",
        "it divides by that sentence's token count, not by your real request size -- at "
        "half the length you get roughly twice the concurrency",
    ),
    "kv_cache.tokens": Gloss(
        "Total tokens the cache can hold, across all sequences.",
        "tokens",
        "tokens are the invariant here; the GiB moves with the dtype and the block layout",
    ),
    "kv_cache.requested_gib": Gloss(
        "The cache size you asked for, when the version supports sizing it directly "
        "(--kv-cache-memory) instead of through a utilization fraction.",
        "GiB",
        "asking is not getting: the engine prints what it actually allocated next to it, "
        "and the difference is what a later short-sequence OOM is about",
    ),
    "kv_cache.non_kv_gib": Gloss(
        "The part of the memory split that is not the KV cache: weights plus the buffers "
        "the engine keeps outside it.",
        "GiB",
        "in the extensible-cache split this is a residual, so it grows when the cache "
        "shrinks without any weight changing",
    ),
    "kv_cache.padding_waste": Gloss(
        "The share of KV memory lost to padding layers added so the hybrid attention "
        "layout fits the block size (kv_cache.padding_layers says how many).",
        "percent of KV memory",
        "it is a cost the engine chose to pay, not a leak, and it is absent on models "
        "that need no padding",
    ),
    "kv_cache.suggested_gib": Gloss(
        "The --kv-cache-memory the engine suggests when the utilization fraction cannot "
        "express what it needs.",
        "GiB",
        "the suggestion is derived from one profile pass; a suggestion quoted in an issue "
        "is that run's number, not a property of the model",
    ),
    "cache": Gloss(
        "Request-level cache behaviour: prefix caching, radix tree, chunked prefill. "
        "These change what the server does, not what it allocated.",
        "a label, or a size",
        "a disabled prefix cache usually costs throughput, not correctness, so it can "
        "hide in a diff that only looks at latency",
    ),
    "mamba": Gloss(
        "State cache for hybrid/Mamba models: its size, and the effective size after the "
        "scheduler raised it to what the decode side needs.",
        "slots, GiB",
        "the effective size can differ per DP/TP/EP rank in one launch, and the cap it "
        "imposes on concurrency is a different key (sched.max_running_requests)",
    ),
    # -- kernels ------------------------------------------------------------------
    "backend": Gloss(
        "Which kernel the build chose for each job -- attention, MoE, linear, all-reduce, "
        "sampling -- and, in backend.candidates, the set it would have considered.",
        "backend names",
        "a chosen backend that did not change can still sit on a list that shrank; that is "
        "why candidates is a fact of its own",
    ),
    "attn": Gloss(
        "The attention backend as the platform printed it when the usual `Using X attention "
        "backend` line never appears (CPU builds); the value is the enum's short name.",
        "backend name",
        "here `default` means no flag won and the platform default did, which is not the "
        "same as an explicitly configured backend",
    ),
    "backend.attention": Gloss(
        "The attention backend the engine selected after its own checks, which is not "
        "always the one the flag asked for (backend.attention_forced says when it was "
        "overridden).",
        "backend name",
        "FlashInfer and FlashAttention are different kernel sets for the same command "
        "line, so this key is the one to read before comparing tokens/s",
    ),
    "backend.moe": Gloss(
        "The MoE backend chosen, with the quantization it runs in when the log names it "
        "(e.g. VLLM_CUTLASS (NvFp4)).",
        "backend name",
        "the same name over a different quantization is a different kernel -- read the "
        "parenthesised part",
    ),
    "moe": Gloss(
        "MoE configuration the engine settled on.",
        "a label",
        "an absent MoE config is not a bug on a dense model; it only matters when the "
        "architecture has experts",
    ),
    "compile": Gloss(
        "torch.compile / inductor on the backbone: backend, ranges compiled, fusions "
        "enabled, seconds spent, and why a recompile happened.",
        "seconds, token ranges",
        "compile.total_s hides which phase grew; compile.dynamo_s, graph_s and warmup_s "
        "split it",
    ),
    "cuda_graph": Gloss(
        "CUDA graph capture: mode, how many graphs, seconds and GiB spent, and the pool "
        "they reserve; cuda_graph.profiling says whether the profile pass includes them.",
        "count, seconds, GiB",
        "capture time is startup time and pool GiB is memory the cache cannot use; two "
        "runs can differ in both and only one of them is a regression",
    ),
    "parallel": Gloss(
        "The parallel layout the process actually runs: TP, PP, DP, world size, the roles "
        "assigned to ranks, and whether TP exceeds the GPUs present.",
        "counts",
        "TP in this block is what ran; arg.tensor_parallel_size is what was asked and can "
        "differ when the engine re-plans",
    ),
    "spec": Gloss(
        "Speculative decoding specifics: draft length, MTP layer reuse and whether the "
        "draft shares embedding/lm_head weights with the target.",
        "tokens, a label",
        "sharing weights changes memory and numerics; a spec diff that only compares "
        "acceptance rates is missing it",
    ),
    "sampling": Gloss(
        "Sampling kernel and overrides: which top-k/top-p implementation ran, whether it "
        "fell back, and which request settings were overridden by spec decoding.",
        "backend names, settings",
        "a silent fallback to the torch kernel is slower, not wrong -- it shows here and "
        "nowhere in the flags",
    ),
    "kernel": Gloss(
        "Kernel-level choices and events: JIT during inference, FlashInfer autotune, and "
        "the final IR op priority per fused operator.",
        "a label or an op name",
        "kernel.ir_op.* is the priority list after platform defaults, so a diff there is "
        "a change in which fused op wins, not in what you asked for",
    ),
    "jit": Gloss(
        "JIT behaviour inside the startup window: warmup seconds, and the mode used for "
        "compiles that happen during inference.",
        "seconds, mode name",
        "a JIT that runs during inference stalls requests instead of startup; that is why "
        "kernel.jit_during_inference exists separately",
    ),
    "quant": Gloss(
        "Quantization as the build reports it: checkpoint format, the kernel selected, and "
        "the per-layer choice when the log lists it.",
        "format and kernel names",
        "a quantized checkpoint that loads is not proof the fast kernel was used; "
        "quant.kernel.selected is",
    ),
    "gdn": Gloss(
        "Gated delta net kernels: which one is used for prefill and which for decode.",
        "kernel names",
        "the two are chosen separately, so a prefill speedup can leave decode untouched",
    ),
    "transfer": Gloss(
        "KV transfer between prefill and decode workers: staging buffers, their size and "
        "method, the custom memory pool, and the NIXL backend.",
        "count, MiB, backend name",
        "staging memory is reserved per worker; a transfer diff that ignores it compares "
        "throughput of two runs with different memory available for the cache",
    ),
    "pd": Gloss(
        "Prefill/decode disaggregation warmup, as a startup step.",
        "a label",
        "its cost lands in startup, before the first request, and no flag accounts for it",
    ),
    "nccl": Gloss(
        "The collective library: version, and any environment overrides that changed how "
        "it behaves.",
        "version, NAME=value",
        "overrides can come from the image rather than the command line, and they decide "
        "which algorithm NCCL picks",
    ),
    "triton": Gloss(
        "Triton driver/library versions as the log prints them.",
        "versions",
        "a mismatch here often shows up first as a compile failure, not as a wrong number",
    ),
    "uva_buffers": Gloss(
        "Whether Unified Virtual Addressing buffers could use pinned memory, and what the "
        "server fell back to.",
        "a label",
        "pinned memory affects host-to-device speed and can change startup timing while "
        "changing no flag",
    ),
    "ray": Gloss(
        "Ray's environment handling: which variable prefixes the workers inherit.",
        "prefixes",
        "a variable missing from this list is missing in the workers, even if the engine "
        "process has it",
    ),
    "cpu": Gloss(
        "CPU-side threading and affinity: torch threads, OMP threads, NUMA mode and "
        "membind, reserved cores, and the core ids bound.",
        "counts, ids",
        "torch_parallelism on a many-core box can be far above the useful number and cost "
        "startup time; it is a real fact, not a typo",
    ),
    "mm": Gloss(
        "Multimodal behaviour: whether the model is effectively text-only, whether the "
        "per-frame pixel cap is applied, the encoder cache budget, chunked mm input, "
        "prefix attention, and the multimodal warmup seconds.",
        "labels, tokens, seconds",
        "mm.text_only means every modality limit was 0, so the vision tower never ran -- "
        "the absence of every other mm.* fact is the evidence, not a reading gap",
    ),
    "torch": Gloss(
        "The CUDA allocator's own OOM report: which device, how much was requested, how "
        "much was free at that instant, the card's total, and torch.oom_holder_gib -- the "
        "GiB another process was holding.",
        "MiB, GiB",
        "oom_holder_gib is the answer to 'VRAM is enough, why did it die': on the A100 in "
        "vllm#54559 another process held 94.41 of 95.00 GiB",
    ),
    # -- serving surface ------------------------------------------------------------
    "arg": Gloss(
        "What the command line asked for. vLLM prints only non-default args, so an absent "
        "arg.X means 'default', not 'unset'; SGLang prints its whole server_args dict. "
        "args.source names the block the values came from.",
        "the flag's own unit",
        "asked is not used: enforce_eager, hybrid dtype rewrites and kv-cache sizing all "
        "override an arg, and the non-arg facts are where the used value lives",
    ),
    "args": Gloss(
        "Which block the args were read from and how complete it is.",
        "a label",
        "vLLM's block is a diff against defaults, SGLang's is the full set -- an arg "
        "comparison across engines is not symmetric",
    ),
    "server": Gloss(
        "The HTTP surface: the URL it binds, the command line it was started with, the "
        "tasks it advertises, and whether scale-out is enabled.",
        "URL, command, labels",
        "the number after `API server` in vLLM's line is a local rank in the DP group, "
        "not a port; the port is only in the URL",
    ),
    "serving": Gloss(
        "How the served endpoint formats content (chat template content format).",
        "a label",
        "it changes what the model receives for multimodal inputs, and no sampling flag "
        "records it",
    ),
    "api": Gloss(
        "Frontend behaviour: tool parser loaded, request fields the server accepted and "
        "then ignored, and chat-template ordering rules it rewrote.",
        "labels, a dict of ignored fields",
        "api.ignored_request_fields is the frontend telling you your request was not "
        "fully honoured; a benchmark that ignores it measures something else",
    ),
    "chat": Gloss(
        "Chat template warmup: whether the template compiled at startup and how long it "
        "took.",
        "a label, milliseconds",
        "a template that warms up here can still fail on a request shape the warmup did "
        "not cover",
    ),
    "plugins": Gloss(
        "Plugin registrations the server made: which name replaced which "
        "(`plugins.loaded: a -> b`).",
        "name pairs",
        "the arrow direction is `registered name -> implementation`, and a missing arrow "
        "is a plugin that never loaded",
    ),
    # -- scheduler and failure ------------------------------------------------------
    "scheduler": Gloss(
        "The scheduler's decisions: async scheduling on or off (and the reason), chunked "
        "prefill, and how it exited.",
        "labels",
        "async scheduling can be accepted from the flag and then disabled by the "
        "executor; scheduler.async_reason is why",
    ),
    "sched": Gloss(
        "Scheduler limits that were changed from what was configured: max scheduled "
        "tokens from spec settings, max running requests capped by a state cache.",
        "counts",
        "sched.max_running_requests carries the cache that caused the cap in its value, "
        "because the number alone reads like a configured limit",
    ),
    "startup": Gloss(
        "How startup ended: ready, or failed and at which stage. startup.error is the "
        "exception text the server printed; startup.cause is a timeout raised around it.",
        "a label, exception text",
        "`stable` and `ready` are different claims: a failed run can still have printed "
        "every fact above it, so read this key first when the diff looks surprising",
    ),
    "startup.error": Gloss(
        "The exception as the server printed it, first line only.",
        "exception text",
        "a wrapper exception (worker init failed, ray task error) names the stage, not the "
        "cause; the cause is further down the same log",
    ),
    "after_ready": Gloss(
        "Errors printed after the server reported ready -- during serving or shutdown.",
        "exception text",
        "these say nothing about startup configuration, which is why they are kept apart "
        "from startup.error",
    ),
}

# The vocabulary itself: every key the bundled corpus produces. Kept here so `explain`
# can tell a key that exists but is only covered by its family from a typo -- the two
# used to look identical, and a typo answered with a family gloss reads as documented.
KNOWN_KEYS = frozenset((
    "after_ready.error",
    "api.chat_template_order",
    "api.ignored_request_fields",
    "api.tool_parser",
    "arg.backend.attention_arg",
    "arg.backend.sampling_arg",
    "arg.cuda_graph.disabled",
    "arg.dtype",
    "arg.gpu_memory_fraction",
    "arg.kv_cache_dtype",
    "arg.max_model_len",
    "arg.model",
    "arg.port",
    "arg.prefix_cache",
    "arg.tensor_parallel_size",
    "args.source",
    "attn.backend",
    "backend.allreduce_dispatch",
    "backend.allreduce_flashinfer",
    "backend.allreduce_fusion",
    "backend.allreduce_potential",
    "backend.allreduce_reason",
    "backend.attention",
    "backend.attention_forced",
    "backend.attention_mm",
    "backend.attention_vit",
    "backend.candidates",
    "backend.flash_attn_blocked",
    "backend.flash_attn_version",
    "backend.gdn_prefill",
    "backend.linear",
    "backend.linear_fallback",
    "backend.moe",
    "backend.moe_prepare_finalize",
    "backend.sampling_flashinfer",
    "cache.chunked_prefill",
    "cache.prefix",
    "cache.radix",
    "chat.template_warmup",
    "compile",
    "compile.artifacts",
    "compile.autotune_gemm",
    "compile.backend",
    "compile.cached_range",
    "compile.deep_gemm",
    "compile.deep_gemm_feature",
    "compile.dynamo_s",
    "compile.fusions",
    "compile.graph_s",
    "compile.inductor_disabled",
    "compile.range",
    "compile.recompile_reason",
    "compile.total_s",
    "compile.warmup_s",
    "cpu.core_ids",
    "cpu.numa_auto",
    "cpu.numa_membind",
    "cpu.numa_migrate",
    "cpu.omp_threads",
    "cpu.reserved_cpus",
    "cpu.set_num_threads",
    "cpu.threads_bind_mode",
    "cpu.torch_parallelism",
    "cpu.torch_threads",
    "cuda_graph",
    "cuda_graph.capture_gib",
    "cuda_graph.capture_s",
    "cuda_graph.captures",
    "cuda_graph.mode",
    "cuda_graph.pool_gib",
    "cuda_graph.profiling",
    "engine",
    "engine.git_sha",
    "engine.init_s",
    "engine.platform",
    "engine.version",
    "env.cuda_driver",
    "env.hip",
    "env.lib.flashinfer_cubin",
    "env.lib.flashinfer_jit_cache",
    "env.lib.interegular",
    "env.lib.modelscope",
    "env.lib.outlines",
    "env.lib.torchao",
    "env.nvcc",
    "env.os",
    "env.pip.flashinfer-python",
    "env.pip.numpy",
    "env.pip.sglang-kernel",
    "env.pip.torch",
    "env.pip.transformers",
    "env.pip.triton",
    "env.python",
    "env.rocm",
    "env.torch",
    "env.torch_rocm",
    "env.ulimit.soft",
    "env.var.CUBLASMP_VERSION",
    "env.var.CUBLAS_VERSION",
    "env.var.CUDA_ARCH_LIST",
    "env.var.CUDA_CACHE_DISABLE",
    "env.var.CUDA_COMPONENT_LIST",
    "env.var.CUDA_DRIVER_VERSION",
    "env.var.CUDA_HOME",
    "env.var.CUDA_MPS_LOG_DIRECTORY",
    "env.var.CUDA_MPS_PIPE_DIRECTORY",
    "env.var.CUDA_VERSION",
    "env.var.CUDNN_FRONTEND_VERSION",
    "env.var.CUDNN_HOME",
    "env.var.CUDNN_VERSION",
    "env.var.I_MPI_HYDRA_BOOTSTRAP",
    "env.var.I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS",
    "env.var.LD_LIBRARY_PATH",
    "env.var.MAX_JOBS",
    "env.var.MKL_NUM_THREADS",
    "env.var.NCCL_IB_DISABLE",
    "env.var.NCCL_IB_GID_INDEX",
    "env.var.NCCL_IB_HCA",
    "env.var.NCCL_IB_QPS_PER_CONNECTION",
    "env.var.NCCL_IB_SL",
    "env.var.NCCL_IB_TC",
    "env.var.NCCL_IB_TIMEOUT",
    "env.var.NCCL_MIN_NCHANNELS",
    "env.var.NCCL_NET_GDR_LEVEL",
    "env.var.NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE",
    "env.var.NCCL_SOCKET_IFNAME",
    "env.var.NCCL_VERSION",
    "env.var.NVIDIA_CTK_LIBCUDA_DIR",
    "env.var.NVIDIA_DRIVER_CAPABILITIES",
    "env.var.NVIDIA_GDRCOPY",
    "env.var.NVIDIA_PRODUCT_NAME",
    "env.var.NVIDIA_REQUIRE_CUDA",
    "env.var.OMP_NUM_THREADS",
    "env.var.PYTORCH_NVML_BASED_CUDA_CHECK",
    "env.var.PYTORCH_ROCM_ARCH",
    "env.var.PYTORCH_TRITON_VERSION",
    "env.var.PYTORCH_VERSION",
    "env.var.TORCHINDUCTOR_CACHE_DIR",
    "env.var.TORCHINDUCTOR_COMPILE_THREADS",
    "env.var.TORCH_CUDA_ARCH_LIST",
    "env.var.VLLM_BUILD_COMMIT",
    "env.var.VLLM_BUILD_PIPELINE",
    "env.var.VLLM_BUILD_URL",
    "env.var.VLLM_CPU_KVCACHE_SPACE",
    "env.var.VLLM_ENABLE_CUDA_COMPATIBILITY",
    "env.var.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
    "env.var.VLLM_IMAGE_TAG",
    "env.var.VLLM_PLUGINS",
    "env.var.VLLM_SERVER_DEV_MODE",
    "env.var.VLLM_USAGE_SOURCE",
    "env.var.VLLM_USE_DEEP_GEMM",
    "env.var.VLLM_USE_FLASHINFER_SAMPLER",
    "env.var.VLLM_WORKER_MULTIPROC_METHOD",
    "env.var.ZE_AFFINITY_MASK",
    "gdn.decode_kernel",
    "gdn.prefill_kernel",
    "gpu.bf16_support",
    "gpu.budget_gib",
    "gpu.compute_capability",
    "gpu.consumed_gib",
    "gpu.cudagraph_gib",
    "gpu.free_gib",
    "gpu.model",
    "gpu.peak_activation_gib",
    "gpu.total_gib",
    "gpu.utilization",
    "gpu.utilization_effective",
    "gpu.utilization_to_match_kv_cache",
    "jit.monitor_mode",
    "jit.warmup_s",
    "kernel.flashinfer_autotune",
    "kernel.ir_op.fused_add_rms_norm",
    "kernel.ir_op.gelu_and_mul_sparse",
    "kernel.ir_op.rms_norm",
    "kernel.jit_during_inference",
    "kv_cache.attention_block_size",
    "kv_cache.cpu_node",
    "kv_cache.dtype",
    "kv_cache.fp8_scaling",
    "kv_cache.group_sizes",
    "kv_cache.layout",
    "kv_cache.lcm_block_sizes",
    "kv_cache.mamba_mode",
    "kv_cache.mamba_page_pad",
    "kv_cache.mamba_ssm_dtype",
    "kv_cache.max_concurrency",
    "kv_cache.max_seq_len",
    "kv_cache.memory_gib",
    "kv_cache.non_kv_gib",
    "kv_cache.padding_layers",
    "kv_cache.padding_waste",
    "kv_cache.requested_gib",
    "kv_cache.suggested_bytes",
    "kv_cache.suggested_gib",
    "kv_cache.tokens",
    "mamba.cache_size",
    "mamba.effective_size",
    "mamba.state_gib",
    "mm.chunked_input",
    "mm.encoder_cache",
    "mm.pixel_cap",
    "mm.prefix_attention",
    "mm.text_only",
    "mm.warmup_s",
    "model.architecture",
    "model.cast",
    "model.max_len",
    "model.tasks",
    "model_init.seconds",
    "model_loading.gib",
    "model_loading.seconds",
    "model_runner",
    "moe.config",
    "nccl.env_overrides",
    "nccl.version",
    "parallel.dp",
    "parallel.dp_leader",
    "parallel.dp_sync_nccl",
    "parallel.pp",
    "parallel.roles",
    "parallel.tp",
    "parallel.tp_exceeds_gpus",
    "parallel.world_size",
    "pd.warmup",
    "plugins.loaded",
    "quant.checkpoint_format",
    "quant.kernel.config",
    "quant.kernel.nvfp4",
    "quant.kernel.selected",
    "ray.env_prefixes",
    "sampling.ignored_with_spec",
    "sampling.overridden",
    "sampling.topk_topp_backend",
    "sampling.topk_topp_fallback",
    "sched.max_num_scheduled_tokens",
    "sched.max_running_requests",
    "scheduler.async",
    "scheduler.async_reason",
    "scheduler.chunked_prefill",
    "scheduler.exit",
    "server.bind",
    "server.command",
    "server.scale_out",
    "server.tasks",
    "server.url",
    "serving.chat_template_content_format",
    "spec.mtp_forward_repeat",
    "spec.mtp_weights_shared",
    "startup",
    "startup.cause",
    "startup.error",
    "torch.oom_device",
    "torch.oom_free_mib",
    "torch.oom_holder_gib",
    "torch.oom_requested_mib",
    "torch.oom_total_gib",
    "transfer.custom_mem_pool",
    "transfer.nixl_backend",
    "transfer.staging_buffer_mib",
    "transfer.staging_buffers",
    "transfer.staging_method",
    "transfer.staging_total_mib",
    "triton.drivers",
    "uva_buffers",
    "weights.available_ram_gib",
    "weights.cache",
    "weights.checkpoint_gib",
    "weights.download_s",
    "weights.dtype",
    "weights.filesystem",
    "weights.format",
    "weights.load_s",
    "weights.model",
    "weights.prefetch",
))

_FAMILY_ORDER = sorted(k for k in GLOSSARY if "." not in k)


def lookup(key: str):
    """Return (matched key or family, Gloss) for `key`, longest prefix first."""
    parts = key.split(".")
    for n in range(len(parts), 0, -1):
        candidate = ".".join(parts[:n])
        if candidate in GLOSSARY:
            return candidate, GLOSSARY[candidate]
    return None, None


def families():
    """The family-level entries, for the bare `explain` listing."""
    return [(k, GLOSSARY[k]) for k in _FAMILY_ORDER if k in GLOSSARY]
