import io
import json
import re
import subprocess
import sys
from pathlib import Path

from servediff import cli as servediff
from servediff import audit as servediff_audit

FIX = Path(__file__).with_name("fixtures")


def facts(name: str) -> dict:
    return servediff.parse((FIX / name).read_text(encoding="utf-8"))["facts"]


def changed(a: str, b: str) -> dict:
    rows = servediff.diff(servediff.parse((FIX / a).read_text()), servediff.parse((FIX / b).read_text()))
    return {r["key"]: (r["baseline"], r["candidate"]) for r in rows if r["status"] != "same"}


# --- Public issue excerpts -------------------------------------------------------

def test_47047_moe_backend_difference_is_found():
    # The difference the reporter found by hand in vllm#47047.
    diff = changed("vllm-47047-v0.23-default.log", "vllm-47047-v0.23-marlin.log")
    assert diff["backend.moe"] == ("VLLM_CUTLASS (NvFp4)", "MARLIN (NvFp4)")
    # The marlin excerpt also quotes the concurrency line the default one omits; that
    # is a real presence difference (reported as absent), not a backend change.
    assert diff["kv_cache.max_concurrency"] == ("<absent>", 54.14)


def test_53472_version_swap_is_found():
    diff = changed("vllm-53472-v0.23.log", "vllm-53472-v0.26.log")
    assert diff["backend.moe"] == ("FlashInfer CUTLASS (Unquantized)", "TrtLlmBf16ExpertsMonolithic")
    # v0.23 excerpt has no prepare/finalize line: absent, not assumed.
    assert diff["backend.moe_prepare_finalize"] == ("<absent>", "MoEPrepareAndFinalizeNoDPEPMonolithic")


def test_candidate_backend_list_is_kept_including_ones_not_chosen():
    # The excerpt lists every MoE backend the build considered. The chosen one can stay
    # the same while this list shrinks, so it is a fact of its own.
    f = facts("vllm-53472-v0.23.log")
    assert f["backend.candidates"].value == "['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']"


def test_53472_flag_restores_a_different_backend_name():
    diff = changed("vllm-53472-v0.26.log", "vllm-53472-v0.26-flag.log")
    assert diff == {
        "backend.moe": ("TrtLlmBf16ExpertsMonolithic", "FlashInferExperts"),
        "backend.moe_prepare_finalize": ("MoEPrepareAndFinalizeNoDPEPMonolithic",
                                         "MoEPrepareAndFinalizeNoDPEPModular"),
    }


def test_evidence_points_at_source_line():
    fact = facts("vllm-53472-v0.26.log")["backend.moe"]
    assert fact.evidence[0][0] == 2
    assert "TrtLlmBf16ExpertsMonolithic" in fact.evidence[0][1]


def test_real_server_log_from_033_is_parsed():
    # First fixture taken from a live server rather than an issue excerpt: vLLM 0.33.0
    # on one RTX 4060 Laptop (8 GB) under WSL2, Qwen2.5-0.5B, captured by `capture`.
    # It guards the whole chain: a real multiprocess log in, the same facts out.
    f = facts("vllm-0.33.0-real-rtx4060.log")
    assert f["engine.version"].value == "0.33.0"
    assert f["model_runner"].value == "V2"
    assert f["backend.attention"].value == "FLASH_ATTN"
    assert f["kv_cache.layout"].value == "LBNHC"
    assert f["uva_buffers"].value == "fallback: device memory"
    assert f["parallel.world_size"].value == 1
    assert f["startup"].value == "ready"
    # Facts an excerpt cannot carry: the graph capture list and the compiled artifacts.
    assert f["cuda_graph.captures"].value == ["PIECEWISE x51", "FULL x2", "FULL x35"]
    assert f["model_loading.gib"].value == 0.93
    # What the engine settled on, and the capacity it ended up with.
    assert f["model.architecture"].value == "Qwen2ForCausalLM"
    assert f["model.max_len"].value == 2048
    assert f["kv_cache.memory_gib"].value == 1.91
    assert f["kv_cache.tokens"].value == 166848
    assert f["kv_cache.max_concurrency"].value == 81.47
    assert f["engine.init_s"].value == 33.06
    assert f["server.bind"].value == "http://0.0.0.0:8001"
    # What the 8 GB card had left, and what the engine was allowed to take.
    assert f["gpu.free_gib"].value == 6.88
    assert f["gpu.total_gib"].value == 8.0
    assert f["gpu.utilization"].value == 0.45
    assert f["gpu.budget_gib"].value == 3.6
    # The same sentence continues into the post-capture breakdown. Consumed memory barely
    # moves between versions; peak activation and the graph reservation are what an upgrade
    # changes without changing a flag, so an OOM report is unreadable without this triple.
    assert f["gpu.consumed_gib"].value == 1.15
    assert f["gpu.peak_activation_gib"].value == 0.53
    assert f["gpu.cudagraph_gib"].value == 0.18
    # Silent gates inside startup: JIT compile time and whether late compiles are loud.
    assert f["jit.warmup_s"].value == 0.0
    assert f["jit.monitor_mode"].value == "warn"
    assert f["compile.autotune_gemm"].value == "disabled (not enough SMs)"
    # torch.compile is one number in the summary line (9.82 s) and three parts here; an
    # upgrade that doubles startup is usually one of these three moving, not all of them.
    assert f["compile.total_s"].value == 9.82
    assert f["compile.dynamo_s"].value == 3.61
    # The log prints the pair as `(1, 2048)`; joining the two captures with a comma alone
    # rendered "1,2048", which reads as one malformed number rather than a range, and it
    # was shipped that way in every table until 0.4.88.
    assert f["compile.range"].value == "(1, 2048)"
    assert f["compile.graph_s"].value == 4.17
    assert f["compile.warmup_s"].value == 0.13
    # Serving torch threads are cut, and the pair is the setting: "1" alone would not say
    # whether the cut came from 16 or from 4.
    assert f["cpu.torch_threads"].value == "16 -> 1"
    # Sampling defaults came from the model, not from the flags. The value is the dict
    # the warning quotes and nothing else: the rule used to read to end of line, so the
    # row carried the log's own follow-up sentence ("If this is not intended, please
    # relaunch ...") and an unbalanced backtick into a markdown table cell.
    assert f["sampling.overridden"].value == ("{'repetition_penalty': 1.1, 'temperature': 0.7,"
                                              " 'top_k': 20, 'top_p': 0.8}")
    # One fact per op: the whole IrOpPriorityConfig line is too wide for the table and
    # two different configs would truncate to the same visible string.
    assert f["kernel.ir_op.rms_norm"].value == "native"
    assert f["kernel.ir_op.gelu_and_mul_sparse"].value == "triton|native"


def test_real_cpu_log_from_034_keeps_the_backend_choices():
    # The corpus' first GPU-less log: vLLM 0.34.0rc1.dev398 on an AVX2 CPU, Qwen3.5-0.8B
    # W8A8, posted in vllm#67598. On this build the GPU-shaped "Using X attention backend"
    # line never prints; before 0.4.62 the whole log came out as `startup ready` with zero
    # backend facts, so a diff of two CPU runs could not see a backend change at all —
    # `<absent>` on both sides, reported as "no difference".
    f = facts("full-cpu-2026-11/vllm-0.34.0rc1-cpu-avx2-qwen3.5.log")
    assert f["backend.attention_vit"].value == "TORCH_SDPA (default)"
    assert f["backend.attention_mm"].value == "TORCH_SDPA"
    assert f["backend.linear"].value == "CPUInt8ScaledMMLinearKernel (CompressedTensorsW8A8Int8)"
    assert f["backend.gdn_prefill"].value == "CPU (head_k_dim=128)"
    # The line that says the fast path was not used. No flag records this fallback, and a
    # benchmark only sees the slower result.
    assert f["backend.linear_fallback"].value == "oneDNN -> torch"
    # The hybrid cache: 0.34 prints per-group block sizes instead of "Available KV cache
    # memory", and mamba mode is a default the model chose, not a flag.
    assert f["kv_cache.group_sizes"].value == "[640, 640, 640, 640]"
    assert f["kv_cache.mamba_mode"].value == "align (default for Qwen3_5ForConditionalGeneration)"
    # Two startup steps that only exist on this kind of run.
    assert f["mm.warmup_s"].value == 9.45
    assert f["compile.recompile_reason"].value == "cache load failure (backbone)"
    # CPU serving: where the worker was pinned, and whether OMP came from outside.
    assert f["cpu.threads_bind_mode"].value == "auto (auto_setup=True, skip_setup=False)"
    assert f["cpu.core_ids"].value == "[1, 3, 5, 7, 9, 11, 13]"
    assert f["cpu.omp_threads"].value == "7 (torch left at 7)"
    # Hybrid cache: the block size is raised so the attention page fits the mamba page.
    assert f["kv_cache.attention_block_size"].value == "640 (page >= mamba page)"
    # Vision tokenisation runs without the reference pixel cap.
    assert f["mm.pixel_cap"].value == "not applied (cap_pixels_per_frame unset)"
    # 0.4.64 read the rest of what this log prints about the deployment. Every one of these
    # had been reachable only as an "unread" line — two of them were even counted as prose
    # noise, which is how a zero remainder can be reported while settings stay hidden.
    assert f["mm.encoder_cache"].value == "16384 tokens, 1 image item(s)"
    assert f["kv_cache.mamba_page_pad"].value == "20.75% (to match attention page)"
    assert f["kv_cache.cpu_node"].value == "4.0/30.69 GiB (node 0)"
    assert f["kv_cache.lcm_block_sizes"].value == "640"
    assert f["compile.inductor_disabled"].value == "user settings"
    assert f["api.chat_template_order"].value == "system-first required (inline system messages rewritten)"
    # NUMA is reported by two layers that disagree, so it stays two facts: the Python layer
    # starts the worker unbound, the C++ layer then binds memory with MEMBIND after a
    # migrate that failed. Collapsing them into one value would have to pick a side.
    assert f["cpu.numa_auto"].value == "unavailable (started unbound)"
    assert f["cpu.numa_membind"].value == "MEMBIND nodes [0]"
    assert f["cpu.numa_migrate"].value == "failed (errno 1)"
    # The backend refuses to move torch threads once the affinity is bound, which is why
    # the thread count you see in the log is not the one you asked for.
    assert f["cpu.set_num_threads"].value == "skipped (thread binding owns affinity)"
    assert f["startup"].value == "ready"
    # The GPU-only fact stays absent rather than being inferred from a CPU default.
    assert "backend.attention" not in f


def test_cpu_backend_choices_are_read_not_offered_back_as_unread():
    # `audit` promises to hand back lines no rule reads, and `parse` must not read a line
    # `audit` still calls unread. These five are the same lines the fixture test asserts on.
    from servediff import audit

    unread = set(audit.unread((FIX / "full-cpu-2026-11/vllm-0.34.0rc1-cpu-avx2-qwen3.5.log").read_text(encoding="utf-8")))
    for line in (
        "(Worker pid=122) INFO 11-14 03:20:19 [interface.py:439] Using default backend AttentionBackendEnum.TORCH_SDPA for vit attention",
        "(Worker pid=122) INFO 11-14 03:20:19 [mm_encoder_attention.py:372] Using AttentionBackendEnum.TORCH_SDPA for MMEncoderAttention.",
        "(Worker pid=122) INFO 11-14 03:20:19 [__init__.py:880] Selected CPUInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8",
        "(Worker pid=122) INFO 11-14 03:20:19 [qwen_gdn_linear_attn.py:360] Using CPU GDN prefill kernel (head_k_dim=128).",
        "(Worker pid=122) WARNING 11-14 03:20:20 [utils.py:885] Failed to create oneDNN linear, fallback to torch linear. Exception: could not create a primitive descriptor for the matmul primitive.",
    ):
        assert audit.shape(line) not in unread, line


def test_no_rule_reads_a_value_to_the_end_of_a_quoted_sentence():
    # `sampling.overridden` captured `(.+)$` and shipped "{...}`. If this is not intended,
    # please relaunch vLLM ..." as a fact value: the log's own advice became part of the
    # value, and its unbalanced backtick split a markdown table cell. A rule that starts
    # at a quoted literal has to end at that literal's closing brace, so every value of
    # this key in the corpus must be a dict a reader could eval, not a sentence.
    import ast
    seen = 0
    for path in sorted(FIX.rglob("*.log")):
        facts = servediff.parse(path.read_text(encoding="utf-8", errors="replace"))["facts"]
        fact = facts.get("sampling.overridden")
        if not fact:
            continue
        for value in fact.values:
            seen += 1
            parsed = ast.literal_eval(value)
            assert "temperature" in parsed, (path.name, value)
            assert "please relaunch" not in value, (path.name, value)
    assert seen >= 1


def test_the_cpu_log_has_nothing_left_worth_a_look(capsys):
    # This log was kept outside the root corpus because four shapes were still unread. 0.4.64
    # read all four, so the check is now the strict one: `--max-shapes 0`, like the root
    # corpus and the held-out directories. What remains is 121 noise lines — routes, banner
    # art, deprecations, progress bars, the broadcast wait — and PROVENANCE.md says why the
    # broadcast line is noise rather than a fact.
    d = FIX / "full-cpu-2026-11"
    assert servediff.main(["audit", str(d), "--max-shapes", "0"]) == 0
    out = capsys.readouterr().out
    assert "nothing worth a look" in out
    # No setting is handed back as noise any more, including the two that used to be.
    for gone in ("NUMA binding", "Padding mamba page size", "Explicitly set (", "kv lcm block sizes"):
        assert gone not in out


def test_a_sentence_a_log_actually_printed_is_never_prose_noise():
    # The audit's "prose" rule exists for text written around a pasted log. Applied to
    # engine output it was a false negative factory: a capitalised English sentence with
    # no colon and a trailing period counted as noise, and `audit` then said
    # "nothing worth a look" while a real setting sat unread underneath. The gate is the
    # log preamble, not the shape of the sentence.
    from servediff import audit

    logged = ("(Worker pid=122) INFO 11-14 03:20:21 [interface.py:1174] Padding mamba page "
              "size by 20.75% to ensure that mamba page size and attention page size are "
              "exactly equal")
    assert audit.is_noise(logged) is False
    assert servediff.parse(logged + "\n")["facts"]["kv_cache.mamba_page_pad"].value == (
        "20.75% (to match attention page)")
    prose = ("This is the full log I get when I start the server with two workers and an "
             "0.9 utilization on a 24 GB card.")
    assert audit.is_noise(prose) is True


def test_the_memory_breakdown_of_a_real_oom_report_is_read():
    # Verbatim lines from vllm#65907 (v0.32.0 @ 0.97), where the whole argument is that
    # consumed memory stayed at ~45 GiB while peak activation dropped 14.16 -> 8.97 GiB.
    # Both line shapes here differ from what the shipped fixtures print: the equivalent-
    # utilization sentence ends right after the number, and the breakdown stands alone.
    text = (
        "INFO 11-05 12:20:12 [gpu_worker.py:774] CUDA graph memory profiling is enabled "
        "(default since v0.21.0). The current --gpu-memory-utilization=0.9700 is equivalent "
        "to --gpu-memory-utilization=0.7715 ...\n"
        "INFO 11-05 12:21:02 [gpu_worker.py:812] Desired GPU memory utilization is "
        "(0.97, 61.39 GiB). Actual usage is 45.05 GiB for consumed memory (weights + "
        "non-torch), 14.16 GiB for peak activation, and 10.32 GiB for CUDAGraph memory. "
        "... Current kv cache memory in use is 2.18 GiB.\n"
    )
    f = servediff.parse(text)["facts"]
    assert f["gpu.utilization_effective"].value == 0.7715
    assert f["gpu.consumed_gib"].value == 45.05
    assert f["gpu.peak_activation_gib"].value == 14.16
    assert f["gpu.cudagraph_gib"].value == 10.32


def test_the_extensible_kv_cache_accounting_of_a_real_report_is_read():
    # Verbatim lines from vllm#66193 (v0.33.x extensible KV cache, 0.85 utilization). The
    # report's headline — "cutting KV 2.4x" — is the ratio of the suggested --kv-cache-memory
    # against the KV memory the error says is available, so both sides must be facts.
    text = (
        "INFO [gpu_worker.py:1213] Free memory on device (30.8/31.46 GiB) on startup. Desired "
        "GPU memory utilization is (0.85, 26.75 GiB). Actual usage is 21.14 GiB for consumed "
        "memory (weights + non-torch), 1.79 GiB for peak activation, and 0.18 GiB for CUDAGraph "
        "memory. Replace gpu_memory_utilization config with `--kv-cache-memory=3746943079` "
        "(3.49 GiB) to fit into requested memory.\n"
        "INFO [extensible_kv_cache.py:598] Memory after warmup: 8.66 GiB free, 0.78 GiB committed "
        "to the KV cache, 3.83 GiB kept for transient activations; 28 KV cache blocks "
        "(1.51 GiB) fit within the requested 26.75 GiB. Torch may reserve up to 28.65 GiB.\n"
        "ERROR [core.py:1707] ValueError: To serve at least one request with the model's max seq "
        "len (36000), (3.35 GiB KV cache is needed, which is larger than the available KV cache "
        "memory (1.46 GiB). Based on the available memory, the estimated maximum model length "
        "is 7488.\n"
    )
    f = servediff.parse(text)["facts"]
    assert f["kv_cache.transient_gib"].value == 3.83
    assert f["kv_cache.after_warmup_free_gib"].value == 8.66
    assert f["kv_cache.blocks"].value == 28
    assert f["kv_cache.blocks_gib"].value == 1.51
    assert f["torch.may_reserve_gib"].value == 28.65
    assert f["kv_cache.suggested_gib"].value == 3.49
    assert f["kv_cache.available_gib"].value == 1.46
    assert f["kv_cache.needed_gib"].value == 3.35
    assert f["model.max_len_fit"].value == 7488
    # The claim, recomputed from facts: 3.49 / 1.46 = 2.39x, and the reservation overshoots
    # the requested budget (28.65 > 26.75) — that is the 5.2 GiB story in another checkout.
    assert round(f["kv_cache.suggested_gib"].value / f["kv_cache.available_gib"].value, 2) == 2.39
    assert f["torch.may_reserve_gib"].value > f["gpu.budget_gib"].value


def test_the_allocator_shortfall_under_a_real_oom_is_read():
    # Verbatim line from vllm#54559 (39 of these under a "No available shared memory
    # broadcast block found in 60 seconds" headline): the allocator is the only place in the
    # log that says how much was asked for and what was actually free at that instant.
    text = (
        "[rank0]:[W831 11:33:26.640394862 CUDACachingAllocator.cpp:3933] memory allocation "
        "failed with OOM on device 0 while trying to allocate 20971520 bytes "
        "(free: 9633792, total: 84974239744).\n"
    )
    f = servediff.parse(text)["facts"]
    assert f["torch.oom_device"].value == 0
    assert f["torch.oom_requested_mib"].value == 20.0
    assert f["torch.oom_free_mib"].value == 9.19
    assert f["torch.oom_total_gib"].value == 79.14
    # The shortfall in the allocator's own units: it wanted 2.2x what was left.
    assert round(f["torch.oom_requested_mib"].value / f["torch.oom_free_mib"].value, 2) == 2.18


def test_the_flags_a_frontend_dropped_and_the_draft_len_it_halved_are_read():
    # Verbatim lines from vllm#58580 (Gemma 4 MTP + speculative decoding on the Rust frontend).
    # The run passes --enable-auto-tool-choice and --structured-outputs-config; the frontend
    # accepts both and says it will ignore them, which is the plainest form of "same flags,
    # different server" — the flag never reached anything.
    text = (
        "(RustFrontend pid=1185) WARNING 09-24 15:00:44 [unsupported.rs:67] argument "
        "'structured_outputs_config' currently has no effect in Rust frontend, ignoring\n"
        "(RustFrontend pid=1185) WARNING 09-24 15:00:44 [unsupported.rs:67] argument "
        "'enable_auto_tool_choice' currently has no effect in Rust frontend, ignoring\n"
        "INFO 09-24 15:00:44 [speculative.py:1220] Overriding draft model max model len "
        "from 262144 to 131072\n"
    )
    f = servediff.parse(text)["facts"]
    assert f["frontend.ignored_arg"].value == [
        "structured_outputs_config", "enable_auto_tool_choice"
    ]
    # Halved, not set: the pair is kept because the upper bound is what the upgrade changed.
    assert f["spec.draft_max_len"].value == "262144 -> 131072"


DROPPED_FLAGS = (
    "(RustFrontend pid=1185) WARNING 09-24 15:00:44 [unsupported.rs:67] argument "
    "'structured_outputs_config' currently has no effect in Rust frontend, ignoring\n"
    "(RustFrontend pid=1185) WARNING 09-24 15:00:44 [unsupported.rs:67] argument "
    "'enable_auto_tool_choice' currently has no effect in Rust frontend, ignoring\n"
)


def test_a_dropped_flag_is_stated_as_a_consequence_not_as_a_row(tmp_path, capsys):
    # `show` lists eighteen facts; the dropped flag is the only one that means the run
    # is not what its command line says. As a plain row it reads as trivia, and two runs
    # that both dropped it diff as `same` — so the verdict has to be printed separately.
    log = tmp_path / "dropped.log"
    log.write_text(
        "INFO 09-24 15:00:40 [api_utils.py:347]  version 0.28.0\n"
        "(RustFrontend pid=1185) WARNING 09-24 15:00:44 [unsupported.rs:67] argument "
        "'enable_auto_tool_choice' currently has no effect in Rust frontend, ignoring\n"
    )
    assert servediff.main(["show", str(log)]) == 0
    out = capsys.readouterr().out
    assert "frontend.ignored_arg" in out  # the row is still there, unchanged
    assert ("the command line named 1 flag(s) this frontend accepted and never applied "
            "(enable_auto_tool_choice): the run is not configured the way the command reads"
            in out)

    # A log with no dropped flag gains no note, so the documented `show` output holds.
    plain = tmp_path / "plain.log"
    plain.write_text("INFO 09-24 15:00:40 [api_utils.py:347]  version 0.28.0\n")
    assert servediff.main(["show", str(plain)]) == 0
    assert "note:" not in capsys.readouterr().out


def test_a_dropped_flag_survives_into_the_diff_notes(tmp_path, capsys):
    a = tmp_path / "a.log"
    a.write_text(DROPPED_FLAGS)
    b = tmp_path / "b.log"
    b.write_text(DROPPED_FLAGS)
    # Both sides dropped the same two flags: every fact is `same`, yet the run is not
    # configured as asked, and the note is the only place that can say so.
    assert servediff.main(["diff", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "0 difference(s)" in out
    assert "baseline: the command line named 2 flag(s)" in out
    assert "candidate: the command line named 2 flag(s)" in out
    assert "structured_outputs_config, enable_auto_tool_choice" in out


# --- Synthetic lines in the formats of the user's real logs -----------------------

VLLM = """\
INFO 10-02 18:10:33 [api_utils.py:347]  version 0.30.0
INFO 10-02 18:10:33 [api_utils.py:286] non-default args: {'port': 8019, 'dtype': 'float16', 'max_model_len': 2048, 'enforce_eager': True, 'served_model_name': ['m'], 'gpu_memory_utilization': 0.45, 'enable_prefix_caching': False}
WARNING 10-02 18:10:33 [vllm.py:1547] Enforce eager set, disabling torch.compile and CUDAGraphs.
INFO 10-02 18:10:53 [cuda.py:538] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER'].
INFO 10-02 18:11:06 [cuda.py:538] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER'].
"""

SGLANG = """\
[2026-10-02 20:22:31] server_args={'model_path': '/m', 'dtype': 'float16', 'context_length': 2048, 'mem_fraction_static': 0.45, 'disable_radix_cache': True, 'disable_cuda_graph': True, 'tp_size': 1, 'attention_backend': 'triton', 'sampling_backend': 'pytorch'}
\x1b[1;33m[2026-10-02 20:22:49] Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled'\x1b[0m
"""


def test_vllm_args_and_repeated_lines():
    parsed = servediff.parse(VLLM)
    f = parsed["facts"]
    assert parsed["engine"] == "vllm"
    assert f["engine.version"].value == "0.30.0"
    assert f["backend.attention"].values == ["FLASH_ATTN"]  # repeated line kept once
    assert f["arg.prefix_cache"].value is False
    assert f["arg.cuda_graph.disabled"].value is True


def test_sglang_shared_args_are_normalised():
    f = servediff.parse(SGLANG)["facts"]
    assert f["arg.prefix_cache"].value is False  # disable_radix_cache=True inverted
    assert f["arg.max_model_len"].value == 2048
    assert f["cuda_graph.prefill"].value == "disabled"  # ANSI codes stripped


def test_sglang_launch_command_is_kept_verbatim():
    # The launch line carries the flags SGLang defaulted, so it is the one place the
    # full invocation survives; args alone cannot show what was left implicit.
    f = facts("sglang-0.4.4-tp4.log")
    command = f["server.command"].value
    assert command.startswith("python3.12 -m sglang.launch_server")
    assert "--tensor-parallel-size 4" in command
    assert "--trust-remote-code" in command


def test_cross_engine_diff_lines_up_shared_facts():
    rows = {r["key"]: r for r in servediff.diff(servediff.parse(VLLM), servediff.parse(SGLANG))}
    assert rows["arg.dtype"]["status"] == "same"
    assert rows["arg.prefix_cache"]["status"] == "same"
    assert rows["engine"]["status"] == "changed"
    assert rows["backend.attention"]["status"] == "only_baseline"
    assert rows["arg.backend.attention_arg"]["status"] == "only_candidate"


FAILED_STARTUP = """\
(EngineCore pid=3811) INFO 10-28 11:26:32 [gpu_worker.py:396] Using V2 Model Runner
(EngineCore pid=3811) ERROR 10-28 11:26:34 [core.py:1346] EngineCore failed to start.
(EngineCore pid=3811) ERROR 10-28 11:26:34 [core.py:1346] RuntimeError: UVA is not available
(APIServer pid=3492) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
"""

FALLBACK_STARTUP = """\
(EngineCore pid=4381) INFO 10-28 11:28:12 [gpu_worker.py:441] Using V2 Model Runner
(EngineCore pid=4381) WARNING 10-28 11:28:15 [buffer_utils.py:47] Pinned memory is not available on this platform; falling back to device memory for UVA buffers.
(APIServer pid=4100) INFO:     Application startup complete.
"""


def test_startup_failure_and_root_error_are_recorded():
    f = servediff.parse(FAILED_STARTUP)["facts"]
    assert f["startup"].value == "failed"
    assert f["startup.error"].values[0] == "RuntimeError: UVA is not available"
    assert f["model_runner"].value == "V2"


def test_uva_fallback_vs_failure_is_a_difference(capsys):
    rows = {r["key"]: r for r in servediff.diff(servediff.parse(FAILED_STARTUP), servediff.parse(FALLBACK_STARTUP))}
    assert rows["startup"]["baseline"] == "failed" and rows["startup"]["candidate"] == "ready"
    assert rows["uva_buffers"]["status"] == "only_candidate"
    assert rows["model_runner"]["status"] == "same"


def test_errors_after_ready_are_not_startup_errors():
    f = servediff.parse(FALLBACK_STARTUP + "RuntimeError: reentrant call inside <_io.BufferedWriter>\n")["facts"]
    assert "startup.error" not in f
    assert f["after_ready.error"].value.startswith("RuntimeError: reentrant call")


def test_failed_side_is_warned_about(tmp_path, capsys):
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(FAILED_STARTUP)
    b.write_text(FALLBACK_STARTUP)
    assert servediff.main(["diff", str(a), str(b)]) == 1
    assert "baseline server failed to start: RuntimeError: UVA is not available" in capsys.readouterr().out


def test_audit_gate_fails_only_when_the_parser_is_behind(tmp_path, capsys):
    # The count comes from the audit module, not from parsing the printed report back:
    # the report truncates shapes, so a text-based count would be wrong on big logs.
    log = tmp_path / "run.log"
    log.write_text("INFO:     Started server process [123]\n"
                   "Plain chatter about the weather, no configuration in it\n")
    assert servediff.main(["audit", str(log), "--max-shapes", "0"]) == 1
    assert "is above --max-shapes 0" in capsys.readouterr().err
    assert servediff.main(["audit", str(log), "--max-shapes", "5"]) == 0
    # Without the flag the exit code stays 0: a report is not a verdict.
    assert servediff.main(["audit", str(log)]) == 0
    # Access logs are noise, not "worth a look", so a threshold of 0 still passes.
    noisy = tmp_path / "noise.log"
    noisy.write_text('INFO:     127.0.0.1:52344 - "GET /health HTTP/1.1" 200 OK\n')
    assert servediff.main(["audit", str(noisy), "--max-shapes", "0"]) == 0


def test_a_cpu_only_collect_env_from_a_real_issue_is_read_and_looks_clean():
    # vllm#66525: a real `collect_env` pasted into an issue, from a box with no GPU at all.
    # Its version lines are unpadded -- the form every bug report actually carries -- and
    # they were unread for five releases because the rules required whitespace before the
    # colon. A hand-copied fixture never showed it.
    f = facts("vllm-66525-cpu-collectenv.log")
    assert f["env.torch"].values == ["2.10.0+cpu"]
    assert f["env.os"].values == ["Debian GNU/Linux 12 (bookworm) (x86_64)"]
    assert f["env.python"].values == ["3.12.15"]
    assert f["gpu.model"].values == ["No CUDA"]
    text = (FIX / "vllm-66525-cpu-collectenv.log").read_text(encoding="utf-8")
    look, _ = servediff_audit.split(*servediff_audit.unread_samples(text))
    assert look == {}, look


def test_the_second_half_of_a_call_is_noise_but_an_env_var_is_not():
    # What was left after 0.4.33's rules: argument lists standing alone, and a continuation
    # that closes parentheses without ever starting a statement. Both come from a script pasted
    # into the issue, so they are the same class as the rest -- but a line that sets a real
    # environment variable must stay readable, because that is a fact.
    script = (
        "8, 96, 128, conv_kernel_size=4, num_spec=5\n"
        "config, groups, num_blocks=5\n"
        "groups, KVCacheLayout.BLHNC\n"
        "x, None, ids, w, routing_input_mode=RoutingInputMode.PackedPrecomputed), wp))(ml)\n")
    look, _ = servediff_audit.split(*servediff_audit.unread_samples(script))
    assert look == {}, look
    # The counter-examples: a `NAME=value` line is a setting, one of them with commas in the
    # value. Device selection is excluded on purpose (GPU numbering is per box), so it is
    # noise and not a fact -- that asymmetry is the existing design, not a miss.
    facts = servediff.parse("NVIDIA_DRIVER_CAPABILITIES=compute,utility\n")["facts"]
    assert facts["env.var.NVIDIA_DRIVER_CAPABILITIES"].values == ["compute,utility"]
    assert not servediff_audit.is_noise("NVIDIA_DRIVER_CAPABILITIES=compute,utility")


def test_audit_json_is_the_same_count_as_the_report(tmp_path, capsys):
    # A CI gate should read fields, not scrape a human layout. The numbers must be the ones
    # the report prints: the same pass counts them.
    log = tmp_path / "run.log"
    log.write_text("Router gate chose expert 3 for every token on this run\n",
                   encoding="utf-8")
    assert servediff.main(["audit", str(log), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["logs_with_shapes_worth_a_look"] == 1
    assert payload["shapes_worth_a_look"] == 1
    entry = payload["logs"][0]
    # The shape, not the line: digits fold to N, because two runs of the same message with
    # different ids are one shape and that is how the report counts them too.
    assert entry["shapes_worth_a_look"] == [
        {"line": "Router gate chose expert N for every token on this run", "count": 1}]
    assert entry["unread"] == 1


def test_audit_json_and_markdown_do_not_mix(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text("Router gate chose expert 3 for every token on this run\n",
                   encoding="utf-8")
    assert servediff.main(["audit", str(log), "--json", "--markdown"]) == 2
    err = capsys.readouterr().err
    assert "pick one" in err


def test_audit_json_still_honours_the_gate(tmp_path):
    # The exit code is the part a pipeline acts on, so --json must not change it.
    log = tmp_path / "run.log"
    log.write_text("Router gate chose expert 3 for every token on this run\n",
                   encoding="utf-8")
    assert servediff.main(["audit", str(log), "--json", "--max-shapes", "0"]) == 1


def test_a_baseline_turns_the_gate_into_a_regression_gate(tmp_path, capsys):
    # A gate on the absolute count fails forever after the first line the parser cannot read,
    # which is how a gate gets switched off. A baseline makes it a question about what is new.
    log = tmp_path / "run.log"
    log.write_text("Router gate chose expert 3 for every token on this run\n",
                   encoding="utf-8")
    base = tmp_path / "baseline.json"
    assert servediff.main(["audit", str(log), "--write-baseline", str(base)]) == 0
    recorded = json.loads(base.read_text(encoding="utf-8"))
    assert list(recorded["shapes"]) == ["Router gate chose expert N for every token on this run"]
    # The line is the same line that fails without a baseline, and the gate passes now.
    assert servediff.main(["audit", str(log), "--baseline", str(base),
                           "--max-shapes", "0"]) == 0
    captured = capsys.readouterr()
    assert "1 shape(s) in the baseline" in captured.err
    # The row is not noise: it is a shape the baseline accepted, and saying "every unread line
    # is noise" here would misdescribe what the tool just did.
    assert "nothing new" in captured.out
    assert "every unread line is noise" not in captured.out


def test_a_new_shape_still_fails_against_a_baseline(tmp_path):
    log = tmp_path / "old.log"
    log.write_text("Router gate chose expert 3 for every token on this run\n", encoding="utf-8")
    base = tmp_path / "baseline.json"
    assert servediff.main(["audit", str(log), "--write-baseline", str(base)]) == 0
    fresh = tmp_path / "new.log"
    fresh.write_text("Router gate chose expert 3 for every token on this run\n"
                     "A line the baseline never saw, about six or more words here\n",
                     encoding="utf-8")
    assert servediff.main(["audit", str(fresh), "--baseline", str(base),
                           "--max-shapes", "0"]) == 1


def test_audit_json_names_what_the_baseline_hid(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text("Router gate chose expert 3 for every token on this run\n", encoding="utf-8")
    base = tmp_path / "baseline.json"
    assert servediff.main(["audit", str(log), "--write-baseline", str(base)]) == 0
    capsys.readouterr()
    assert servediff.main(["audit", str(log), "--baseline", str(base), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shapes_worth_a_look"] == 0
    assert payload["baseline_shapes_not_counted"] == 1
    assert payload["baseline_lines_not_counted"] == 1


def test_a_file_that_is_not_a_baseline_is_an_input_error(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text("Router gate chose expert 3 for every token on this run\n", encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text('{"lines": []}\n', encoding="utf-8")
    assert servediff.main(["audit", str(log), "--baseline", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "not a baseline" in err


def test_a_second_month_of_pastes_finds_nothing_either(capsys):
    # The batch collected one month after the first, so this is what tests whether the noise
    # table learned October or learned logs. At 0.4.36 one line survived here: the conditional
    # expression `run(a) if a.cmd == "run" else compare(a)` in sglang-49403.log.
    held = FIX / "held-out-2026-11"
    logs = sorted(held.glob("*.log"))
    assert len(logs) == 7, logs
    assert servediff.main(["audit", str(held)]) == 0
    out = capsys.readouterr().out
    assert f"{len(logs)} log(s): 0 with shapes worth a look" in out


def test_a_conditional_expression_standing_alone_is_noise(capsys):
    text = "run(a) if a.cmd == \"run\" else compare(a)"
    look, _ = servediff_audit.split(*servediff_audit.unread_samples(text))
    assert look == {}
    # The same shape does not swallow a real warning that happens to contain both keywords.
    real = ("Using FlashInfer all-reduce if available else falling back to the custom kernel "
            "(set VLLM_ALLREDUCE_USE_FLASHINFER=0 to disable)")
    look2, _ = servediff_audit.split(*servediff_audit.unread_samples(real))
    assert look2 != {}, look2


def test_a_third_month_of_pastes_finds_nothing_either(capsys):
    # The batch collected on 11-10 with created:>2026-11-05, which is a different question from
    # the two earlier directories: those asked whether the rules had learned October, this one
    # asks whether they had learned the first week of November. At 0.4.38 it reported 17 of 28
    # files, 191 shapes, almost all of them the `[conda]` package inventory that four reports
    # pasted along with collect_env.
    held = FIX / "held-out-2026-11b"
    logs = sorted(held.glob("*.log"))
    assert len(logs) == 28, logs
    assert servediff.main(["audit", str(held)]) == 0
    out = capsys.readouterr().out
    assert f"{len(logs)} log(s): 0 with shapes worth a look" in out


def test_a_conda_inventory_row_is_noise(capsys):
    # collect_env lists every installed wheel twice, once as `[pip3] name==version` and once as
    # a conda table row. The pip form was read for the few wheels that decide kernels; the table
    # row says the same thing about the same wheel, so it is not a shape worth a look.
    text = ("[conda] torch                     2.13.0+cu130             pypi_0    pypi\n"
            "[conda] nvidia-cublas             13.1.1.3                 pypi_0    pypi\n")
    look, _ = servediff_audit.split(*servediff_audit.unread_samples(text))
    assert look == {}, look


def test_a_path_expression_is_noise_but_an_enginecore_line_is_not(capsys):
    # A pasted reproducer builds its output path with an expression and then calls a method on
    # it. The wider `^\(.*\)$` was measured first and rejected: it ate 31 real vLLM lines,
    # every one of them a `(EngineCore pid=…)` entry. That line is the control here: audit reads
    # it as a log entry, so it must not turn up as a shape worth a look either.
    engine = ("(EngineCore pid=4381) INFO 10-28 11:30:24 [core.py:372] init engine"
              " (profile, create kv cache) done")
    text = "(r / 'pressure-native-inputs.json').write_text(json.dumps({'a': b}))\n" + engine + "\n"
    look, noise = servediff_audit.split(*servediff_audit.unread_samples(text))
    # The expression is classified as noise; the engine line is not, which is the whole point of
    # the narrow pattern. Whether audit has a *fact* rule for the engine line is a different
    # question, and this control line is invented, so it is not asserted here.
    assert any("pressure-native-inputs" in shape for shape in noise), noise
    assert not any("pressure-native-inputs" in shape for shape in look), look
    assert not servediff_audit.is_noise(engine), engine


def test_a_pasted_call_chain_is_noise(capsys):
    # The other script shapes one batch brought: an f-string inside a call, a chained call
    # closed twice on one line, and an argument list that merely closes the call above it.
    text = ("model.load_weights([(f\"{prefix}.{names[0]}\", torch.tensor(0.25))])\n"
            "tf.AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).save_pretrained(path)\n"
            "layer.weight_global_scale, layer.workspace, N, K)\n"
            "async def main(checkout):\n")
    look, _ = servediff_audit.split(*servediff_audit.unread_samples(text))
    assert look == {}, look


def test_the_held_out_corpus_reports_nothing_to_look_at(capsys):
    # Eleven pastes taken fresh from vLLM and SGLang issues in October 2026, shipped so the
    # claim in writeups/audit-noise-real-logs.md can be rerun instead of believed. At v0.4.32
    # this command reported 8 of 11 files with lines worth a look.
    held = FIX / "held-out-2026-10"
    logs = sorted(held.glob("*.log"))
    assert len(logs) == 11, logs
    assert servediff.main(["audit", str(held)]) == 0
    out = capsys.readouterr().out
    assert f"{len(logs)} log(s): 0 with shapes worth a look" in out


def test_a_directory_is_audited_as_one_corpus_with_a_line_per_log(capsys):
    # The claim "the whole corpus from public reports, nothing worth a look" should be
    # checkable in one command rather than eighty. rglob, because the corpus is kept in
    # dated subdirectories and a one-level glob quietly audited less than a quarter of it.
    logs = sorted(FIX.rglob("*.log"))
    assert servediff.main(["audit", str(FIX)]) == 0
    out = capsys.readouterr().out
    assert f"{len(logs)} log(s): 0 with shapes worth a look" in out
    assert "nothing worth a look" in out
    assert servediff.main(["audit", str(FIX), "--max-shapes", "0"]) == 0


def test_a_corpus_names_the_one_log_that_still_has_a_shape(tmp_path, capsys):
    clean = tmp_path / "clean.log"
    clean.write_text((FIX / "vllm-0.30.0-gvisor-cuda-collectenv.log").read_text(encoding="utf-8"),
                     encoding="utf-8")
    odd = tmp_path / "odd.log"
    odd.write_text("TENSOR_PARALLEL_STRIDE 37 for rank 3 on device 5\n", encoding="utf-8")
    assert servediff.main(["audit", str(tmp_path), "--max-shapes", "0"]) == 1
    out = capsys.readouterr().out
    assert "odd.log" in out
    assert "1 with shapes worth a look" in out


def test_a_library_dependency_dump_is_noise_but_a_missing_library_is_a_fact():
    # vllm#65873 on a GB10 box: `ldd` lists every cudnn module by absolute path. That is a
    # dependency dump, not a setting, and it should not read as thirty "settings".
    assert servediff_audit.is_noise("/usr/lib/aarch64-linux-gnu/libcudnn_ops.so.9.9.9")
    assert servediff_audit.is_noise("/usr/lib/aarch64-linux-gnu/libcudnn_engines_precompiled.so.9.9.9")
    # An ImportError naming a missing library is the opposite case and stays readable.
    assert not servediff_audit.is_noise(
        "ImportError: libcudart.so.13: cannot open shared object file: No such file or directory")


def test_a_gb10_box_reads_its_model_and_its_dependency_list_as_noise():
    path = FIX / "vllm-0.33.0-gb10-nemotron-nvfp4.log"
    f = servediff.parse(path.read_text(encoding="utf-8"))["facts"]
    assert f["engine.version"].value.startswith("0.33.0")
    assert f["gpu.model"].value == "NVIDIA GB10"
    assert f["env.os"].value == "Ubuntu 24.04.5 LTS (aarch64)"
    counts, samples = servediff_audit.unread_samples(path.read_text(encoding="utf-8"))
    look, _ = servediff_audit.split(counts, samples)
    assert not look, sorted(look)


def test_a_per_worker_measurement_is_one_number_from_the_lowest_rank(tmp_path):
    # Eight TP workers each log the same measurement with their own jitter. One box should
    # read as one number, taken from the lowest rank, so two runs of the same box can agree.
    ranks = [(2, "36.339381"), (3, "36.384071"), (7, "36.395200"), (0, "36.312437")]
    lines = [f"(Worker_TP{rank} pid=2717{rank}) INFO 07-14 16:41:00 [model_runner.py:302]"
             f" Model loading took 1.14 GiB and {sec} seconds" for rank, sec in ranks]
    log = tmp_path / "tp.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    f = servediff.parse(log.read_text(encoding="utf-8"))["facts"]
    assert f["model_loading.seconds"].value == 36.312437
    assert f["model_loading.gib"].value == 1.14


def test_two_measurements_from_the_same_rank_are_both_kept(tmp_path):
    # TP0 loading a backbone and a draft head prints twice; that is two measurements,
    # not two workers, and collapsing them to one would hide the second.
    log = tmp_path / "two-loads.log"
    log.write_text(
        "(Worker_TP0 pid=2717792) INFO 07-14 16:40:53 [default_loader.py:430]"
        " Loading weights took 28.64 seconds\n"
        "(Worker_TP0 pid=2717792) INFO 07-14 16:40:59 [default_loader.py:430]"
        " Loading weights took 5.66 seconds\n", encoding="utf-8")
    f = servediff.parse(log.read_text(encoding="utf-8"))["facts"]
    assert f["weights.load_s"].value == [28.64, 5.66]


def test_a_numeric_change_reports_its_magnitude(tmp_path, capsys):
    # "changed" alone reads the same at 4% and at 400%; the magnitude is what a reader
    # decides on.
    a = tmp_path / "a.log"
    a.write_text("(EngineCore pid=1) INFO 10-28 11:36:31 [monitor.py:53]"
                 " torch.compile took 32.58 s in total\n", encoding="utf-8")
    b = tmp_path / "b.log"
    b.write_text("(EngineCore pid=1) INFO 10-28 11:29:11 [monitor.py:53]"
                 " torch.compile took 31.08 s in total\n", encoding="utf-8")
    assert servediff.main(["diff", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "compile.total_s" in out
    assert "changed  -4.6%" in out
    rows = servediff.diff(servediff.parse(a.read_text()), servediff.parse(b.read_text()))
    assert [r["status"] for r in rows if r["key"] == "compile.total_s"] == ["changed"]


def test_a_text_change_has_no_magnitude(tmp_path, capsys):
    a = tmp_path / "a.log"
    a.write_text("Initializing a V1 LLM engine (v0.28.0) with config\n", encoding="utf-8")
    b = tmp_path / "b.log"
    b.write_text("Initializing a V1 LLM engine (v0.30.0) with config\n", encoding="utf-8")
    servediff.main(["diff", str(a), str(b)])
    out = capsys.readouterr().out
    assert "engine.version" in out
    assert "%" not in out


def test_a_reproducer_script_pasted_in_an_issue_is_noise(tmp_path):
    # sglang#49123 pastes a 195-line benchmark script instead of a log. Read as log lines it
    # produced 38 "worth a look" shapes, every one of them the script itself.
    log = tmp_path / "script.log"
    log.write_text(
        '"""One MoE layer through two paths.\n'
        "usage: marlin_vs_cutile.py <model dir> <layer> [tokens]\n"
        '"""\n'
        "import json\n"
        "mm, L = sys.argv[1], int(sys.argv[2])\n"
        "# ---- float32 reference\n"
        "def deq(b, s, g):\n"
        "    return v * (s / g)\n"
        "    if \"W4A4\" in name:\n"
        "handles[f] = safe_open(os.path.join(mm, f), \"pt\")\n"
        "DQ[e] = (deq(t[\"w1\"][0][e], t[\"w1\"][1][e]),\n"
        "print(f\"tokens {m} {name}\", flush=True)\n"
        "MAX_MODEL_LEN=4096\n", encoding="utf-8")
    text = log.read_text(encoding="utf-8")
    counts, samples = servediff_audit.unread_samples(text)
    look, _ = servediff_audit.split(counts, samples)
    shapes = set(look.values())
    assert "def deq(b, s, g):" not in shapes
    assert "handles[f] = safe_open(os.path.join(mm, f), \"pt\")" not in shapes
    assert "mm, L = sys.argv[1], int(sys.argv[2])" not in shapes
    # A shell-style assignment is still read as a fact, so it is neither noise nor a
    # "worth a look" -- the script rules must not swallow it.
    assert "env.var.MAX_MODEL_LEN" in servediff.parse(text)["facts"]
    assert sum(look.values()) == 0, "every line of the script is read as a fact or as noise"


def test_a_loop_condition_with_a_quoted_string_is_noise(tmp_path):
    log = tmp_path / "script.log"
    log.write_text('  if "W4A4" in name:\n', encoding="utf-8")
    text = log.read_text(encoding="utf-8")
    look, _ = servediff_audit.split(*servediff_audit.unread_samples(text))
    assert look == {}


def test_a_collect_env_block_is_read_with_or_without_padding(tmp_path, capsys):
    # vLLM's own `collect_env` prints `PyTorch version: …` with no padding; the ROCm dump
    # aligns the colons with spaces. The rules required the spaces, so they were dead on the
    # real output -- found by auditing a fresh CPU-only issue paste.
    plain = tmp_path / "plain.log"
    plain.write_text(
        "PyTorch version: 2.10.0+cpu\n"
        "OS: Debian GNU/Linux 12 (bookworm) (x86_64)\n"
        "Python version: 3.12.15 (main, Oct 21 2026, 01:08:07) [GCC 12.2.0] (64-bit runtime)\n"
        "HIP runtime version: N/A\n"
        "Caching allocator config: N/A\n"
        "GPU models and configuration: No CUDA\n", encoding="utf-8")
    facts = servediff.parse(plain.read_text(encoding="utf-8"))["facts"]
    assert facts["env.torch"].values == ["2.10.0+cpu"]
    assert facts["env.os"].values == ["Debian GNU/Linux 12 (bookworm) (x86_64)"]
    assert facts["env.python"].values == ["3.12.15"]
    # `N/A` is absent, not a value: it must not become a fact that reads as a difference.
    assert "env.hip" not in facts
    assert "env.allocator" not in facts
    # `No CUDA` is a statement, not an absent value: a CPU box says so, and two CPU boxes
    # agree on it.
    assert facts["gpu.model"].values == ["No CUDA"]

    padded = tmp_path / "padded.log"
    padded.write_text(
        "PyTorch version              : 2.13.0+rocm10.1.0\n"
        "OS                           : Ubuntu 24.04.5 LTS (x86_64)\n", encoding="utf-8")
    facts = servediff.parse(padded.read_text(encoding="utf-8"))["facts"]
    assert facts["env.torch"].values == ["2.13.0+rocm10.1.0"]
    assert facts["env.os"].values == ["Ubuntu 24.04.5 LTS (x86_64)"]


def test_a_macos_environment_report_is_the_machine_not_the_run(tmp_path):
    log = tmp_path / "mac.log"
    log.write_text(
        "MPS available: True\n"
        "macOS Version: 26.7\n"
        "macOS Build: 25G229\n"
        "Apple Silicon: Apple M5\n"
        "Unified Memory: 16.0 GB\n"
        "CPU Cores (Total): 10\n"
        "CPU Cores (Performance): 4\n"
        "Metal Support: Metal 4\n"
        "GPU Cores: 10\n"
        "GPU 0,1,2,3,4,5,6,7: NVIDIA H20-Ne\n"
        "CPU Affinity\tNUMA Affinity\tGPU NUMA ID\n"
        "(16-NIC topology matrix omitted; single GPU, not relevant to this bug)\n"
        "WTYPE = scalar_types.uint8b128\n"
        "tensor.offset\n"
        "{\n"
        "(Path(folder) / \"config.json\").write_text(\n", encoding="utf-8")
    text = log.read_text(encoding="utf-8")
    look, _ = servediff_audit.split(*servediff_audit.unread_samples(text))
    assert look == {}, look


def test_skip_absent_keeps_only_rows_both_logs_printed(tmp_path, capsys):
    # A run that died leaves every later fact <absent>; those one-sided rows outnumber the
    # real differences and bury them. --skip-absent must hide them while still failing, and
    # the counts must not silently reclassify a hidden difference as "identical".
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(FAILED_STARTUP)
    b.write_text(FALLBACK_STARTUP)
    assert servediff.main(["diff", str(a), str(b), "--skip-absent"]) == 1
    out = capsys.readouterr().out
    assert "uva_buffers" not in out, out
    assert "hidden by --skip-absent" in out
    assert "1 difference(s); 1 fact(s) identical." in out, out
    assert "2 difference(s) skipped because one side never printed those lines" in out
    # JSON must stay honest about what it withheld.
    capsys.readouterr()
    assert servediff.main(["diff", str(a), str(b), "--skip-absent", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["differences"] == 3 and payload["compared_both_sides"] == 1
    assert payload["hidden_absent"] == 2
    assert [r["key"] for r in payload["rows"]] == ["startup"]


def test_cuda_graph_captures_and_pool():
    text = (
        "Capturing CUDA graphs (decode, FULL):   0%|          | 0/35 [00:00<?, ?it/s]"
        "Capturing CUDA graphs (decode, FULL):   3%|▎         | 1/35 [00:00<00:04,  7.27it/s]\n"
        "Capturing CUDA graphs (PIECEWISE):   0%|          | 0/51 [00:00<?, ?it/s]\n"
        "Capturing CUDA graphs (PIECEWISE):   0%|          | 0/51 [00:00<?, ?it/s]\n"
        "INFO 10-28 11:37:03 [gpu_worker.py:741] CUDA graph pool memory: 0.36 GiB (actual), 0.38 GiB (estimated)\n"
    )
    f = servediff.parse(text)["facts"]
    assert f["cuda_graph.captures"].values == ["decode, FULL x35", "PIECEWISE x51"]
    assert f["cuda_graph.pool_gib"].value == 0.36


def test_conflicting_values_are_all_kept():
    text = "Using FLASH_ATTN attention backend\nUsing TRITON_ATTN attention backend\n"
    assert servediff.parse(text)["facts"]["backend.attention"].value == ["FLASH_ATTN", "TRITON_ATTN"]


def test_unrecognised_log_exits_2(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("hello\n")
    assert servediff.main(["show", str(log)]) == 2


def test_unreadable_path_is_exit_2_not_a_difference(tmp_path, capsys):
    """A typo'd path must not borrow exit 1, which means "the two sides differ"."""
    missing = str(tmp_path / "not-there.log")
    assert servediff.main(["show", missing]) == 2
    assert servediff.main(["diff", missing, str(tmp_path / "also-missing.log")]) == 2
    err = capsys.readouterr().err
    assert "not-there.log" in err


def test_diff_exit_codes(tmp_path):
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(VLLM)
    b.write_text(VLLM)
    assert servediff.main(["diff", str(a), str(b)]) == 0
    b.write_text(VLLM.replace("FLASH_ATTN attention", "FLASHINFER attention"))
    assert servediff.main(["diff", str(a), str(b)]) == 1


def test_markdown_report_contains_table_evidence_and_failure_warning(tmp_path, capsys):
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(FAILED_STARTUP)
    b.write_text(FALLBACK_STARTUP)
    assert servediff.main(["diff", str(a), str(b), "--markdown"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("# servediff report")
    assert "| Fact | Baseline | Candidate | Status |" in out
    assert "| `startup` |" in out
    assert "⚠️ baseline server failed to start" in out
    assert "<details>" in out and "<summary>Evidence</summary>" in out
    assert "UVA is not available" in out  # quoted evidence line


def test_capture_tees_output_and_writes_env_snapshot(tmp_path):
    log = tmp_path / "run.log"
    code = "import sys; print('hello-from-server'); print('err-line', file=sys.stderr)"
    rc = servediff.main(["capture", "-o", str(log), "--", sys.executable, "-c", code])
    assert rc == 0
    written = log.read_text(encoding="utf-8")
    assert "hello-from-server" in written
    assert "err-line" in written  # stderr merged into the log
    env_file = log.with_suffix(log.suffix + ".env.json")
    assert env_file.exists()
    data = json.loads(env_file.read_text(encoding="utf-8"))
    assert data["schema"] == "servediff-env/1"
    assert "pytest" in data["packages"]


def test_capture_snapshots_the_python_the_command_names(tmp_path, capsys):
    """`capture -- python -m vllm...` must snapshot that interpreter, not servediff's own."""
    log = tmp_path / "run.log"
    rc = servediff.main(["capture", "-o", str(log), "--", sys.executable, "-c", "print('hi')"])
    assert rc == 0
    data = json.loads(log.with_suffix(log.suffix + ".env.json").read_text(encoding="utf-8"))
    assert data["executable"] == sys.executable
    assert "does not name a python launcher" not in capsys.readouterr().err


def test_capture_survives_ctrl_c_and_keeps_the_log(tmp_path, capsys, monkeypatch):
    """Ctrl-C is how a server run ends; exit 1 there would read as "the inputs differ"."""
    log = tmp_path / "run.log"
    terminated = []

    class FakeProc:
        returncode = None

        def __init__(self):
            self.stdout = self._lines()

        def _lines(self):
            yield "INFO: started serving\n"
            raise KeyboardInterrupt

        def poll(self):
            return None if not terminated else 0

        def terminate(self):
            terminated.append("terminate")

        def wait(self, timeout=None):
            self.returncode = -15

    # Stub the snapshot so the only Popen in play is the fake server's.
    monkeypatch.setattr(servediff.envsnap, "snapshot",
                        lambda *a, **k: {"schema": "servediff-env/1", "packages": {}, "env": {},
                                         "executable": "stub-python"})
    monkeypatch.setattr(servediff.subprocess, "Popen", lambda *a, **k: FakeProc())
    rc = servediff.main(["capture", "-o", str(log), "--", "fake-server"])
    assert rc == 130
    assert terminated == ["terminate"]
    assert "INFO: started serving" in log.read_text(encoding="utf-8")
    assert "interrupted; server stopped, log kept" in capsys.readouterr().err


def test_capture_reports_missing_command(tmp_path, capsys):
    log = tmp_path / "x.log"
    assert servediff.main(["capture", "-o", str(log), "--", "no-such-binary-xyz"]) == 2
    assert "cannot start" in capsys.readouterr().err


TP4_EP = """\
(EngineCore pid=100) INFO 10-28 11:28:12 [parallel_state.py:1827] world_size=4 rank=2 local_rank=2 distributed_init_method=file:///tmp/vllm_dist_x backend=nccl
(EngineCore pid=100) INFO 10-28 11:28:12 [parallel_state.py:2267] rank 2 in world size 4 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 2, ETP rank 2, EP rank 1, EPLB rank 0
(EngineCore pid=100) INFO 10-28 11:28:34 [utils.py:320] Using LBNHC KV cache layout.
"""


def test_tp4_ep_parallel_roles_are_parsed():
    f = servediff.parse(TP4_EP)["facts"]
    assert f["parallel.world_size"].value == 4
    assert f["parallel.roles"].value == ("DP rank 0, PP rank 0, PCP rank 0, TP rank 2, "
                                         "ETP rank 2, EP rank 1, EPLB rank 0")
    assert f["kv_cache.layout"].value == "LBNHC"


def test_model_loading_and_compile_facts():
    text = (
        "(EngineCore pid=100) INFO 10-28 11:28:34 [model_runner.py:428] Model loading took 0.93 GiB memory and 19.122619 seconds\n"
        "(EngineCore pid=100) INFO 10-28 11:29:11 [backends.py:920] collected artifacts: 25 entries, 3 artifacts, 4996920 bytes total\n"
        "(EngineCore pid=100) INFO 10-28 11:29:11 [monitor.py:53] torch.compile took 31.08 s in total\n")
    f = servediff.parse(text)["facts"]
    assert f["model_loading.gib"].value == 0.93
    assert abs(f["model_loading.seconds"].value - 19.122619) < 1e-6
    assert f["compile.artifacts"].value == "25 entries, 3 artifacts, 4996920 bytes"
    assert abs(f["compile.total_s"].value - 31.08) < 1e-9


def test_v1_runner_detected_from_gpu_model_runner_module():
    text = ("(EngineCore pid=100) INFO 10-28 11:35:38 "
            "[gpu_model_runner.py:5419] Starting to load model /models/Qwen\n")
    f = servediff.parse(text)["facts"]
    assert f["model_runner"].value == "V1"


def test_weight_loading_utilization_and_server_surface_facts():
    # Lines the audit called out on a real log, each of which changes what a run costs
    # or what it serves without changing any command-line flag.
    text = (
        "(EngineCore pid=1) INFO 11-08 10:55:29 [weight_utils.py:1144] Filesystem type for"
        " checkpoints: EXT4. Checkpoint size: 0.92 GiB. Available RAM: 2.75 GiB.\n"
        "(EngineCore pid=1) INFO 11-08 10:55:29 [weight_utils.py:1167] Auto-prefetch is disabled"
        " because the filesystem (EXT4) is not a recognized network FS (NFS/Lustre).\n"
        "(EngineCore pid=1) INFO 11-08 10:55:54 [gpu_worker.py:787] CUDA graph memory profiling is"
        " enabled (default since v0.21.0). The current --gpu-memory-utilization=0.4500 is equivalent"
        " to --gpu-memory-utilization=0.4141 without CUDA graph memory profiling. To maintain the"
        " same effective KV cache size as before, increase --gpu-memory-utilization to 0.4859.\n"
        "(EngineCore pid=1) INFO 11-08 10:55:54 [kernel_warmup.py:270] Skipping FlashInfer autotune"
        " because it is disabled.\n"
        "(APIServer pid=2) INFO 11-08 10:56:07 [entry.py:135] Supported tasks: ['generate']\n"
        "(APIServer pid=2) INFO 11-08 10:56:08 [factories.py:105] Scale-out endpoints are disabled."
        " Restart vllm serve with --enable-scale-out to enable them.\n"
        "(APIServer pid=2) INFO 11-08 10:56:08 [hf.py:736] Detected the chat template content"
        " format to be 'string'.\n")
    f = servediff.parse(text)["facts"]
    assert f["weights.filesystem"].value == "EXT4"
    assert f["weights.checkpoint_gib"].value == 0.92
    assert f["weights.available_ram_gib"].value == 2.75
    assert f["weights.prefetch"].value == "disabled (EXT4 is not a network FS)"
    assert f["cuda_graph.profiling"].value == "enabled (default since v0.21.0)"
    # The sentence ends with a full stop; a greedy character class once swallowed it and
    # float() raised on "0.4859." — keep the two numbers apart from the punctuation.
    assert f["gpu.utilization_effective"].value == 0.4141
    assert f["gpu.utilization_to_match_kv_cache"].value == 0.4859
    assert f["kernel.flashinfer_autotune"].value == "disabled"
    assert f["server.tasks"].value == "['generate']"
    assert f["server.scale_out"].value == "disabled"
    assert f["serving.chat_template_content_format"].value == "string"



# Real TP>1 log excerpt, distilled from vLLM issue #48591 (0.23.1, TP=8,
# dspark spec decode crash): every rank prints its own world_size line, the
# assigned-as line has no ETP field, and the run fails at CUDA graph time.
TP8 = """\
(Worker pid=100) INFO 07-14 16:40:12 [parallel_state.py:1607] world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:53729 backend=nccl
(Worker pid=101) INFO 07-14 16:40:13 [parallel_state.py:1607] world_size=8 rank=1 local_rank=1 distributed_init_method=tcp://127.0.0.1:53729 backend=nccl
(Worker pid=102) INFO 07-14 16:40:13 [parallel_state.py:1607] world_size=8 rank=7 local_rank=7 distributed_init_method=tcp://127.0.0.1:53729 backend=nccl
(Worker pid=103) INFO 07-14 16:40:23 [parallel_state.py:1942] rank 0 in world size 8 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(Worker pid=104) INFO 07-14 16:40:23 [gpu_worker.py:378] Using V2 Model Runner
INFO 07-14 16:38:11 [compilation.py:312] Enabled custom fusions: allreduce_rms
(Worker_TP7 pid=105) INFO 07-14 16:40:24 [model_runner.py:281] Loading model from scratch...
(Worker_TP7 pid=106) ERROR 07-14 16:40:44 [model_runner.py:999] RuntimeError: Engine core initialization failed. See root cause above.
"""


def test_tp8_multiple_worker_lines_parse_parallel_config():
    f = servediff.parse(TP8)["facts"]
    assert f["parallel.world_size"].value == 8
    assert f["parallel.roles"].value == ("DP rank 0, PP rank 0, PCP rank 0, TP rank 0, "
                                         "EP rank N/A, EPLB rank N/A")
    assert f["compile.fusions"].value == "allreduce_rms"
    assert f["model_runner"].value == "V2"
    assert f["startup"].value == "failed"


def test_roles_without_etp_field_and_kv_layout_absent():
    text = ("(EngineCore pid=3811) INFO 10-28 11:26:32 [parallel_state.py:1638] "
            "world_size=1 rank=0 local_rank=0 distributed_init_method=file:///tmp/vllm_dist_x backend=nccl\n"
            "(EngineCore pid=3811) INFO 10-28 11:26:32 [parallel_state.py:1982] "
            "rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, "
            "TP rank 0, EP rank N/A, EPLB rank N/A\n")
    f = servediff.parse(text)["facts"]
    assert f["parallel.world_size"].value == 1
    assert "ETP" not in f["parallel.roles"].value  # 0.28 format lacks the field
    assert "kv_cache.layout" not in f


# --- Real SGLang startup log, distilled from sgl-project/sglang#4862 ---------------
# SGLang 0.4.4, TP=4, Qwen2.5-Omni-7B. It prints `server_args=ServerArgs(...)`
# (keyword-call form, not a dict literal); the parser must handle both forms.

def test_sglang_server_args_call_form_is_parsed():
    f = servediff.parse((FIX / "sglang-0.4.4-tp4.log").read_text(encoding="utf-8"))["facts"]
    assert f["engine"].value == "sglang"
    assert f["arg.tensor_parallel_size"].value == 4
    assert f["arg.model"].value == "/home/model/Qwen/Qwen2.5-Omni-7B"
    assert f["arg.gpu_memory_fraction"].value == 0.85
    assert f["arg.port"].value == 30000
    assert f["arg.backend.attention_arg"].value == "flashinfer"
    assert f["args.source"].value == "sglang server_args (all args)"


def test_sglang_startup_failure_is_recorded():
    f = servediff.parse((FIX / "sglang-0.4.4-tp4.log").read_text(encoding="utf-8"))["facts"]
    assert f["startup"].value == "failed"
    assert f["startup.error"].value.startswith("AttributeError:")


def test_cli_version_flag():
    out = subprocess.run([sys.executable, "-m", "servediff", "--version"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    # The version moves with every release, so pin the shape, not the number.
    assert re.fullmatch(r"servediff \d+\.\d+\.\d+\n", out.stdout)


def test_demo_diffs_the_logs_it_ships():
    # A fresh install has no logs, so the first thing a new user runs has to work
    # without any: demo reads two excerpts that travel inside the package.
    out = subprocess.run([sys.executable, "-m", "servediff", "demo"],
                         capture_output=True, text=True)
    assert out.returncode == 1  # the bundled pair really differs
    assert "backend.moe" in out.stdout
    assert "MARLIN" in out.stdout
    assert "demo: exit 1" in out.stderr  # says why 1 is a result, not a failure
    # The demo also has to show the command a hang needs, on two shipped logs, and the
    # subtraction that separates them: one wait began before capture ended, the other began
    # the second a kernel compiled.
    assert "two different readings" in out.stdout
    assert "began at 09:02:37" in out.stdout
    assert "began at 22:01:40" in out.stdout


def test_a_reader_that_stops_early_is_not_a_traceback():
    # `servediff show server.log | head` is how a long fact list gets read. The reader
    # closes the pipe, our write fails, and the interpreter used to print a traceback
    # over the facts the user was reading. `head` itself is not used here: Windows
    # does not ship one, and the failure mode differs there anyway (see the unit test
    # below — Linux raises BrokenPipeError, Windows reports OSError 22 / winerror 232).
    log = Path(__file__).parent / "fixtures" / "vllm-forum-2026-06" / "vllm-cpu-2x-epyc-9655.log"
    reader = 'import sys; sys.stdout.write(sys.stdin.readline()[:400])'
    cmd = f'{sys.executable} -m servediff show "{log}" | {sys.executable} -c "{reader}"'
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out
    assert "cpu.core_ids" in out.stdout
    assert "BrokenPipeError" not in out.stderr
    assert "Traceback" not in out.stderr
    assert "OSError" not in out.stderr


def test_a_windows_style_closed_pipe_is_not_a_traceback(monkeypatch, capsys):
    # Windows never raises BrokenPipeError for this: the CRT reports `OSError: [Errno 22]
    # Invalid argument`. The guard has to recognise that form too, or the same command
    # that is silent on Linux prints a traceback on Windows.
    from servediff import cli

    class ClosedPipe:
        def write(self, _text):
            raise OSError(22, "Invalid argument")

        def flush(self):
            raise OSError(22, "Invalid argument")

        def fileno(self):
            return 1

        def reconfigure(self, **_kwargs):
            pass

        def isatty(self):
            return False

        @property
        def encoding(self):
            return "utf-8"

    log = Path(__file__).parent / "fixtures" / "vllm-forum-2026-06" / "vllm-cpu-2x-epyc-9655.log"
    monkeypatch.setattr(cli.sys, "stdout", ClosedPipe())
    assert cli.main(["show", str(log)]) == 0
    assert "Traceback" not in capsys.readouterr().err


def test_audit_names_the_lines_nothing_reads():
    # The report is what an external user pastes when a fact comes out <absent>.
    from servediff import audit

    real = (FIX / "vllm-0.33.0-real-rtx4060.log").read_text(encoding="utf-8")
    counts = audit.unread(real)
    blocks = audit.report(Path("real.log"), real)
    assert sum(counts.values()) > 0
    assert "unread line(s)" in blocks
    # A fixture every line of which is read says so, instead of printing an empty list.
    excerpt = (FIX / "vllm-47047-v0.23-default.log").read_text(encoding="utf-8")
    assert not audit.unread(excerpt)
    assert "nothing unread" in audit.report(Path("excerpt.log"), excerpt)


def test_audit_separates_noise_from_shapes_worth_a_look():
    # A report that dumps 80 lines, 59 of them route tables and banner art, does not get
    # read. Noise is counted apart, and --everything is the only way to see it — except
    # when nothing is worth a look, where the noise list is the only content there is.
    from servediff import audit

    mixed = ("INFO 11-08 [backends.py:1] A setting no rule knows about: 42\n"
             + 'INFO:     127.0.0.1:8000 - "GET /health HTTP/1.1" 200 OK\n')
    text = audit.report(Path("mixed.log"), mixed)
    assert "1 line(s) worth a look" in text
    assert "no rule knows about" in text
    assert "line(s) of noise" in text
    assert "--everything to list them" in text
    assert "Route: /" not in text and "GET /health" not in text
    assert "GET /health" in audit.report(Path("mixed.log"), mixed, everything=True)

    real = (FIX / "vllm-0.33.0-real-rtx4060.log").read_text(encoding="utf-8")
    counts, samples = audit.unread_samples(real)
    look, noise = audit.split(counts, samples)
    assert sum(noise.values()) > 0
    default = audit.report(Path("real.log"), real)
    assert "line(s) of noise" in default
    assert "Route: /" in default  # nothing worth a look, so the ignored lines are shown
    # Classifying on the shape alone loses the status code, so access logs would stay in
    # the "worth a look" list; they must be noise.
    assert audit.is_noise('INFO:     127.0.0.1:8000 - "GET /health HTTP/1.1" 200 OK')


def test_the_one_real_log_has_nothing_left_worth_a_look():
    # The fixture is a full live-server log, not an excerpt, so "every unread line is
    # noise" is the standard this parser is held to on the only log it has that a stranger
    # could also produce. This fails the day a version adds a configuration line the
    # parser does not read — which is exactly the regression the audit exists to catch.
    from servediff import audit

    real = (FIX / "vllm-0.33.0-real-rtx4060.log").read_text(encoding="utf-8")
    look, _ = audit.split(*audit.unread_samples(real))
    assert not look, sorted(look)  # names the shapes, so the failure says what was missed
    assert "nothing worth a look" in audit.report(Path("real.log"), real)


def test_a_crashed_real_log_does_not_report_its_own_traceback():
    # The most common real log is one that died. Its traceback is dozens of lines, each
    # prefixed with `(EngineCore pid=…) ERROR … [core.py:…]`, and the noise patterns only
    # describe the message — so without stripping that preamble every frame reads as a
    # shape worth a look and the report becomes a stack trace nobody asked for.
    from servediff import audit

    crash = FIX / "vllm-crash-traceback.log"
    text = crash.read_text(encoding="utf-8")
    look, noise = audit.split(*audit.unread_samples(text))
    assert not look, sorted(look)
    assert sum(noise.values()) > 0
    assert "nothing worth a look" in audit.report(crash, text)
    # The exception itself is the one line that must not be buried: it is read by the
    # startup.error rule, so it is not unread at all.
    assert audit.shape('(EngineCore pid=3811) ERROR 10-28 11:26:34 [core.py:1346] '
                       'RuntimeError: UVA is not available') not in audit.unread(text)

    # Stripping must not swallow a real setting that happens to carry the same preamble.
    assert audit.is_noise("(EngineCore pid=3811) Process EngineCore:")
    assert audit.is_noise("(EngineCore pid=3811)   File \"/opt/vllm-env/x.py\", line 34, in __init__")
    assert audit.is_noise("(APIServer pid=4080) INFO 10-28 11:30:34 [launcher.py:129] "
                          "[shutdown] API server: shutdown triggered")
    for setting in (
        "(EngineCore pid=1) INFO 10-28 11:00:00 [cuda.py:118] Using AttentionBackendEnum.FLASHINFER backend.",
        "(EngineCore pid=1) INFO 10-28 11:00:00 [scheduler.py:1] Chunked prefill is enabled with max_num_batched_tokens=2048.",
        "(Worker_TP2 pid=9) INFO 10-28 11:00:00 [model_runner.py:399] Model loading took 1.2 GiB and 1.2 seconds",
    ):
        assert not audit.is_noise(setting), setting


def test_audit_does_not_call_a_read_line_unread():
    # audit and parse() must agree: a line parse() records a value from (here the
    # graph-capture summary) may never be offered back to the user as ignored.
    from servediff import audit
    from servediff.cli import parse

    real = (FIX / "vllm-0.33.0-real-rtx4060.log").read_text(encoding="utf-8")
    facts = parse(real)["facts"]
    assert facts["cuda_graph.capture_s"].value == [3, 4]
    assert facts["cuda_graph.capture_gib"].value == [0.25, 0.18]

    text_lines = real.splitlines()
    counts, samples = audit.unread_samples(real)
    for fact in facts.values():
        for lineno, evidence in fact.evidence:
            if not lineno:
                continue  # engine / arg-derived facts carry no line number
            assert audit.shape(evidence) not in counts, (
                f"line {lineno} is read but reported unread: {evidence[:60]}")


def test_loading_phase_is_three_separate_steps():
    # "Startup got slower" is unanswerable until the phase is split: building the module,
    # reading shards, compiling. These lines were unread until the audit named them.
    text = (
        "(EngineCore pid=1) INFO 11-08 10:55:26 [model_runner.py:399] Loading model from scratch...\n"
        "(EngineCore pid=1) INFO 11-08 10:55:29 [base_loader.py:82] Initializing model took 2.732282 seconds\n"
        "(EngineCore pid=1) INFO 11-08 10:55:32 [default_loader.py:506] Loading weights took 3.13 seconds\n")
    f = servediff.parse(text)["facts"]
    assert f["weights.cache"].value == "miss (from scratch)"
    assert abs(f["model_init.seconds"].value - 2.732282) < 1e-9
    assert f["weights.load_s"].value == 3.13


def test_the_shapes_left_in_a_crashed_log_are_read_not_ignored():
    # After the crash traceback became noise, three shapes were still "worth a look" on the
    # real 0.28.0 logs. None was noise: a coerced dtype, an unimportable optional kernel
    # library, and an output handler that died. Classifying them away would have hidden real
    # conditions, so they become facts instead.
    text = (
        "(APIServer pid=4080) WARNING 10-28 11:26:00 [model.py:1] Casting torch.bfloat16 to torch.float32.\n"
        "(EngineCore pid=3811) WARNING 10-28 11:26:00 [import_utils.py:1] Module deep_gemm was found but failed to import\n"
        "(EngineCore pid=3811) WARNING 10-28 11:26:00 [import_utils.py:1] ImportError: libnvrtc.so.3: cannot open shared object file: No such file or directory\n"
        "(EngineCore pid=3811) WARNING 10-28 11:26:00 [import_utils.py:1] AssertionError\n"
        "(APIServer pid=4080) ERROR 10-28 11:30:00 [async_llm.py:1] AsyncLLM output_handler failed.\n"
        "(APIServer pid=4080) ERROR 10-28 11:30:00 [async_llm.py:1]     outputs = await engine_core.get_output_async()\n"
        "(APIServer pid=4080) ERROR 10-28 11:30:00 [async_llm.py:1] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above)\n")
    f = servediff.parse(text)["facts"]
    assert f["model.cast"].value == "torch.bfloat16 -> torch.float32"
    assert f["kernel.module_missing"].value == "deep_gemm"
    assert f["kernel.missing_lib"].value == "libnvrtc.so.3"
    # `startup` collects every verdict the log printed, and a startup.error also adds the
    # bare "failed" — so membership, not equality.
    assert "failed (output handler)" in f["startup"].value
    assert "failed" in f["startup"].value
    # The exception type arrives with its module path, which the rule must not require to be
    # at line start.
    assert f["startup.error"].value == ("EngineDeadError: EngineCore encountered an issue. "
                                        "See stack trace (above)")
    look, noise = servediff_audit.split(*servediff_audit.unread_samples(text))
    assert not look, sorted(look)
    # The bare AssertionError is the import check above it, already read: counting it again
    # would report one condition twice.
    assert servediff_audit.is_noise("    AssertionError")


def test_a_rocm_collect_env_block_is_read_as_versions_not_as_180_settings():
    # A public ROCm bug report (vllm#65034): the pasted `collect_env` block is ~180 lines and
    # every one of them reads like a setting, so the audit called all of them worth a look.
    # The version lines are facts; the machine inventory inside the same block is not.
    path = FIX / "vllm-0.29.1-rocm-collectenv.log"
    f = servediff.parse(path.read_text(encoding="utf-8"))["facts"]
    assert f["engine.version"].value == "0.29.1.dev0+g98dff2a81.d20261002"
    assert f["engine.git_sha"].value == "98dff2a81"
    assert f["env.torch"].value == "2.13.0+rocm10.1.0"
    assert f["env.rocm"].value == "7.16.26385-0000000"
    assert f["env.hip"].value == "7.16.26385"
    assert f["env.pip.triton"].value == "3.8.0+git669b31ac.rocm10.1.0"
    assert f["gpu.model"].value == "AMD Radeon AI PRO R9700 (gfx1201)"
    assert f["env.var.PYTORCH_ROCM_ARCH"].value.startswith("gfx90a;gfx942")
    # `CMake version : version 3.31.10` must not be mistaken for the engine version.
    assert "3.31.10" not in f["engine.version"].value
    counts, samples = servediff_audit.unread_samples(path.read_text(encoding="utf-8"))
    look, _ = servediff_audit.split(counts, samples)
    assert not look, sorted(look)


def test_a_cuda_collect_env_block_is_read_as_facts_not_as_a_pip_inventory():
    # A production-image report (vllm#63505): the same collect_env block on CUDA. The
    # versions that decide kernels are facts; the ~35 `[pip3] nvidia-*==` lines are an
    # inventory, and "Could not collect" is an absence rather than a value.
    path = FIX / "vllm-0.32.0-cuda-collectenv.log"
    f = servediff.parse(path.read_text(encoding="utf-8"))["facts"]
    assert f["engine.version"].value == "0.32.0"
    assert f["env.var.CUDA_VERSION"].value == "13.0.2"
    assert f["env.var.TORCH_CUDA_ARCH_LIST"].value == "7.5 8.0 8.6 8.9 9.0 10.0 12.0"
    assert f["env.var.VLLM_IMAGE_TAG"].value.startswith("vllm/vllm-openai:v5df63a72")
    assert f["env.var.VLLM_USAGE_SOURCE"].value == "production-docker-image"
    # A CUDA box printed `ROCM Version : Could not collect`, which used to become
    # env.rocm = "Could" and then compare equal to another CUDA box.
    assert "env.rocm" not in f
    assert "env.hip" not in f
    assert not [k for k in f if k.startswith("env.pip.") and k.count("-") > 1]
    counts, samples = servediff_audit.unread_samples(path.read_text(encoding="utf-8"))
    look, _ = servediff_audit.split(counts, samples)
    assert not look, sorted(look)


def test_an_eight_gpu_rocm_log_reads_its_topology_tables_as_noise():
    # vllm#65611: eight MI355X cards, so `collect_env` ends with the rocm-smi weight / hops /
    # link-type matrices and the GPU topology grid. Those print the machine, not the run.
    path = FIX / "vllm-0.33.0rc1-rocm-mi355x.log"
    f = servediff.parse(path.read_text(encoding="utf-8"))["facts"]
    assert f["engine.version"].value == "0.33.0rc1.dev490+gade358752"
    assert f["gpu.model"].value == "AMD Instinct MI355X (gfx950:sramecc+:xnack-)"
    assert f["env.var.PYTORCH_ROCM_ARCH"].value == "gfx90a;gfx942;gfx950"
    assert f["env.torch_rocm"].value == "7.15.26333"
    counts, samples = servediff_audit.unread_samples(path.read_text(encoding="utf-8"))
    look, _ = servediff_audit.split(counts, samples)
    assert not look, sorted(look)


def test_a_sglang_environment_dump_is_read_as_versions_not_as_a_package_list():
    # sgl-project/sglang#41654: SGLang prints `name: value` per line for ~35 packages. The
    # ones that decide which kernels load are facts; the rest is inventory.
    path = FIX / "sglang-0.5.20-env-dump.log"
    f = servediff.parse(path.read_text(encoding="utf-8"))["facts"]
    assert f["engine.version"].value == "0.5.20"
    assert f["env.torch"].value == "2.13.0+cu130"
    assert f["env.pip.sglang-kernel"].value == "0.4.7"
    assert f["env.pip.flashinfer-python"].value == "0.6.18"
    assert f["env.var.CUDA_HOME"].value == "/usr/local/cuda"
    assert f["env.cuda_driver"].value == "580.126.09"
    assert f["env.ulimit.soft"].value == 1048576
    counts, samples = servediff_audit.unread_samples(path.read_text(encoding="utf-8"))
    look, _ = servediff_audit.split(counts, samples)
    assert not look, sorted(look)


def test_a_bracketed_sglang_timestamp_is_a_prefix_not_content():
    # sgl-project/sglang#42684: `[2026-10-06 00:15:19] NIXL KVManager initialized with
    # backend: UCX` — a scheduler that died at startup, on a line whose prefix used to
    # swallow the message. None of it is in `server_args`.
    path = FIX / "sglang-0.5-nixl-startup-failure.log"
    f = servediff.parse(path.read_text(encoding="utf-8"))["facts"]
    assert f["transfer.nixl_backend"].value == "UCX"
    assert f["cache.radix"].value.startswith("source=default impl=UnifiedRadixCache")
    started = f["startup"].value
    assert "failed (scheduler)" in (started if isinstance(started, list) else [started])
    assert f["startup.error"].value.startswith("TypeError: NixlKVManager")
    counts, samples = servediff_audit.unread_samples(path.read_text(encoding="utf-8"))
    look, _ = servediff_audit.split(counts, samples)
    assert not look, sorted(look)


def test_a_per_request_crash_reads_its_indexerror_as_the_cause():
    # sgl-project/sglang#48502: the server starts, then every request raises
    # `IndexError: index is out of bounds for dimension with size 0`.
    path = FIX / "sglang-0.5-indexerror-per-request.log"
    f = servediff.parse(path.read_text(encoding="utf-8"))["facts"]
    assert f["startup"].value == "failed"
    assert f["startup.error"].value.startswith("IndexError: index is out of bounds")
    counts, samples = servediff_audit.unread_samples(path.read_text(encoding="utf-8"))
    look, _ = servediff_audit.split(counts, samples)
    assert not look, sorted(look)


def test_a_multi_node_box_reads_its_per_card_model_and_its_nic_grid_as_noise():
    # vllm#66192, a GB200 NVL node: the aggregate line is printed empty and each card
    # carries the model, and `collect_env` ends with the InfiniBand NIC topology grid.
    # An empty value is an absence, so the promoting pass is what fills gpu.model.
    path = FIX / "vllm-0.33.0rc1-gb200.log"
    f = servediff.parse(path.read_text(encoding="utf-8"))["facts"]
    assert f["gpu.model"].value == "NVIDIA GB200"
    assert all(str(v).strip() for fact in f.values() for v in fact.values)
    assert f["env.os"].value == "Ubuntu 24.04.3 LTS (aarch64)"
    counts, samples = servediff_audit.unread_samples(path.read_text(encoding="utf-8"))
    look, _ = servediff_audit.split(counts, samples)
    assert not look, sorted(look)
    # The topologies and their legends describe the machine, not the run.
    assert servediff_audit.is_noise("NIC0: mlx5_0")
    assert servediff_audit.is_noise("NIC0\tNODE\t X \tPIX\tNODE\tNODE\tNODE\tNODE")
    assert servediff_audit.is_noise("NIC Legend:")


def test_a_single_gpu_report_reads_the_model_once_even_when_the_line_repeats_it():
    # vllm#66193 prints `GPU models and configuration : GPU 0: NVIDIA RTX PRO 6000 …` —
    # the value carries the per-card prefix, which is not part of the model name.
    path = FIX / "vllm-0.33.0rc1-rtx-pro-6000.log"
    f = servediff.parse(path.read_text(encoding="utf-8"))["facts"]
    assert f["gpu.model"].value == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    counts, samples = servediff_audit.unread_samples(path.read_text(encoding="utf-8"))
    look, _ = servediff_audit.split(counts, samples)
    assert not look, sorted(look)


def test_a_stalled_startup_reports_how_long_it_waited():
    # The most repeated line in a stalled startup was classified as noise: 49 copies of
    # "No available shared memory broadcast block found in 60 seconds" in a log that then
    # started fine. The count and the total are the only numbers a reader has for 49
    # minutes of boot, so they are facts, not a note -- and `startup` must still read
    # `ready`, because waiting is not failing.
    log = FIX / "full-cpu-2026-11/vllm-0.34.0rc1-cpu-avx2-qwen3.5.log"
    facts = servediff.parse(log.read_text(encoding="utf-8", errors="replace"))["facts"]
    assert facts["shm.broadcast_stalls"].value == 49
    assert facts["shm.broadcast_wait_s"].value == 2940
    assert facts["startup"].value == "ready"
    # One interval per line, and the first of them is where the evidence points.
    assert facts["shm.broadcast_stalls"].evidence[0][0] == 64
    assert "broadcast block found in 60 seconds" in facts["shm.broadcast_stalls"].evidence[0][1]
    # A line that produces a fact must not come back from audit as unread.
    counts, _ = servediff_audit.unread_samples(log.read_text(encoding="utf-8", errors="replace"))
    shapes = [shape for shape in counts if "shared memory broadcast" in shape]
    assert not shapes, shapes


def test_a_stalled_startup_states_the_total_in_minutes_not_only_two_rows(capsys):
    # Two fact rows eleven seconds apart in a reader's head are not the answer to "why
    # did this take fifty minutes". The product is stated in the report, with the startup
    # outcome attached, so a 5-minute wait and a 49-minute one do not read alike -- and
    # the same sentence has to survive into the markdown report, which is the artifact
    # that gets pasted into an issue.
    log = FIX / "full-cpu-2026-11/vllm-0.34.0rc1-cpu-avx2-qwen3.5.log"
    assert servediff.main(["show", str(log)]) == 0
    out = capsys.readouterr().out
    note = [line for line in out.splitlines() if "waited 2940 s" in line][0]
    assert "49 min" in note and "49 interval(s)" in note and "`startup` reported ready" in note
    assert servediff.main(["show", str(log), "--markdown"]) == 0
    md = capsys.readouterr().out
    assert "> ℹ️ this startup waited 2940 s (49 min)" in md
    # A log that never waited must not grow a sentence about waiting.
    quiet = FIX / "sglang-2026-10/sglang-42684.log"
    assert servediff.main(["show", str(quiet)]) == 0
    assert "waited" not in capsys.readouterr().out


def test_the_tp8_report_log_leaves_nothing_unread_because_these_facts_are_read():    # Every line below is a shape the audit named as "no rule reads them" on a real
    # vLLM 0.23.1 TP=8 startup log. They are the configuration that made that run
    # different, so each one has to become a fact rather than stay in a list of
    # sixty shapes nobody acts on.
    text = (
        "(EngineCore pid=1) INFO 11-07 09:12:01 [vllm.py:13] Asynchronous scheduling is enabled.\n"
        "(EngineCore pid=1) INFO 11-07 09:12:01 [scheduler.py:12] Chunked prefill is enabled with max_num_batched_tokens=16384.\n"
        "(EngineCore pid=1) INFO 11-07 09:12:02 [multiproc_executor.py:16] We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'.\n"
        "(EngineCore pid=1) INFO 11-07 09:12:02 [multiproc_executor.py:18] Reducing Torch parallelism from 96 threads to 1 to avoid unnecessary CPU contention.\n"
        "(Worker_TP0 pid=7) INFO 11-07 09:12:05 [cuda.py:76] Using AttentionBackendEnum.FLASHINFER backend.\n"
        "(Worker_TP0 pid=7) INFO 11-07 09:12:05 [eagle3_utils.py:55] Using Eagle3 auxiliary layers from config: (2, 10, 18, 26, 34)\n"
        "(Worker_TP0 pid=7) INFO 11-07 09:12:40 [model_runner.py:91] Model loading took 1.14 GiB and 36.34 seconds\n"
        "(Worker_TP0 pid=7) INFO 11-07 09:12:44 [weight_utils.py:50] Filesystem type for checkpoints: FUSE.OFS. Checkpoint size: 7.49 GiB. Available RAM: 1411.31 GiB.\n"
        "(Worker_TP0 pid=7) INFO 11-07 09:12:44 [weight_utils.py:51] Auto-prefetch is disabled because the filesystem (FUSE.OFS) is not a recognized network FS (NFS/Lustre/etc).\n"
        "(Worker_TP0 pid=7) INFO 11-07 09:12:50 [backends.py:110] Cache the graph of compile range (1, 102) for later use\n"
        "(Worker_TP0 pid=7) INFO 11-07 09:12:51 [topk_topp_sampler.py:95] Using FlashInfer for top-p & top-k sampling.\n"
        "(Worker_TP0 pid=7) INFO 11-07 09:12:51 [flashinfer.py:128] FlashInfer resolved query dtypes: prefill=torch.bfloat16, decode=torch.bfloat16, decode_backend=xqa,\n"
        "(Worker_TP0 pid=7) INFO 11-07 09:12:52 [pynccl.py:36] vLLM is using nccl==2.28.9\n"
        "(EngineCore pid=1) INFO 11-07 09:12:02 [multiproc_executor.py:19] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=10.0.0.1, mq_connect_ip=10.0.0.1\n"
        "CUDA error (/workspace/.deps/vllm-flash-attn-src/hopper/flash_api.cpp:1): invalid configuration argument\n"
        "(EngineCore pid=1) ERROR 11-07 09:20:01 [multiproc_executor.py:1] Worker proc VllmWorker-0 died unexpectedly (exit code: None), shutting down executor.\n")
    f = servediff.parse(text)["facts"]
    assert f["scheduler.async"].value == "enabled"
    assert f["scheduler.chunked_prefill"].value == "enabled (max_num_batched_tokens=16384)"
    assert f["worker.mp_method"].value == "spawn"
    assert f["cpu.torch_parallelism"].value == "96 -> 1"
    assert f["attn.backend"].value == "FLASHINFER"
    assert f["spec_decode.eagle_layers"].value.startswith("Eagle3: (2, 10, 18, 26, 34)")
    # 0.23 printed this line without the word `memory`; a rule pinned to the newer
    # wording lost the whole fact on the older log instead of reporting it as absent.
    assert f["model_loading.gib"].value == 1.14
    assert f["model_loading.seconds"].value == 36.34
    # FUSE.OFS has a dot; `\w+` stopped at `FUSE` and the rest of the line stopped matching.
    assert f["weights.filesystem"].value == "FUSE.OFS"
    assert f["weights.prefetch"].value == "disabled (FUSE.OFS is not a network FS)"
    assert f["compile.cached_range"].value == "(1, 102)"
    assert f["sampling.topk_topp_backend"].value == "flashinfer"
    assert "decode_backend=xqa" in f["kernel.flashinfer_dtypes"].value
    assert f["nccl.version"].value == "2.28.9"
    assert f["parallel.dp_leader"].value == "node 0 / dp 0"
    # A kernel error and a dead worker are outcomes, not settings to be ignored.
    assert f["kernel.cuda_error"].value.startswith("invalid configuration argument")
    assert f["startup"].value == "failed (worker died)"
    # And nothing here may be offered back to the user as an unread line.
    look, _ = servediff_audit.split(*servediff_audit.unread_samples(text))
    assert not look, sorted(look)


def test_audit_markdown_wraps_one_pasteable_block(tmp_path, capsys):
    # The issue form wants this report as its third field; --markdown makes that a paste.
    # The fence must be longer than any backtick run inside, or the block ends early and
    # the reader silently loses the lines that follow.
    log = tmp_path / "run.log"
    log.write_text('info: server says "use --max-model-len ```8192```" and nothing else\n'
                   "(EngineCore pid=1) INFO 11-08 10:55:26 [x.py:9] Some unread config line\n",
                   encoding="utf-8")
    assert servediff.main(["audit", str(log), "--markdown"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("````text\n"), out[:40]
    assert out.rstrip().endswith("````")
    assert "Some unread config line" in out


def test_package_version_is_the_same_number_the_cli_prints():
    # __version__ used to be a literal "0.1.0" while --version read the installed
    # distribution, so anyone checking servediff.__version__ saw a stale release.
    import servediff as pkg

    assert pkg.__version__ == servediff._version()


def test_allreduce_backend_selection_is_recorded():
    # Real lines from a vLLM 0.23.1 TP=8 startup log (vllm#48591). The chosen
    # all-reduce list and the FlashInfer path are the only positive evidence that
    # a fused all-reduce backend actually ran, so they must not be dropped.
    f = servediff.parse((FIX / "vllm-0.23.1-tp8-allreduce.log").read_text(encoding="utf-8"))["facts"]
    assert f["model_runner"].value == "V2"
    assert f["backend.allreduce_dispatch"].value == "tp:0: ['CUSTOM', 'SYMM_MEM', 'PYNCCL']"
    assert "FLASHINFER" in f["backend.allreduce_potential"].value
    assert f["backend.allreduce_flashinfer"].value == "mnnvl"
    assert f["backend.allreduce_fusion"].values == ["norm: mnnvl", "norm quantization: trtllm"]


def test_multi_group_allreduce_is_kept_per_group():
    # Verbatim lines from a real vLLM 0.29-era 4xH200 TP4+EP startup log pasted in
    # vllm#45604. Two groups mean the fact carries one value per group; a parser
    # that kept only the last line would silently drop the EP side, which is the
    # half that stayed unchanged in vllm#63705.
    f = servediff.parse((FIX / "vllm-0.29-tp4-ep-h200-allreduce.log").read_text(encoding="utf-8"))["facts"]
    assert f["backend.allreduce_dispatch"].values == ["tp:0: ['CUSTOM', 'PYNCCL']", "ep:0: ['PYNCCL']"]
    assert "FLASHINFER" in f["backend.allreduce_potential"].value
    assert f["backend.allreduce_flashinfer"].value == "trtllm"
    assert f["backend.allreduce_fusion"].value == "norm: trtllm"


def test_show_reads_a_log_from_stdin(monkeypatch, capsys):
    # Logs live on the box that runs the server; `ssh node 'cat serve.log' | servediff show -`
    # should not need an scp first.
    raw = (FIX / "vllm-0.29-tp4-ep-h200-allreduce.log").read_bytes()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"))
    assert servediff.main(["show", "-"]) == 0
    out = capsys.readouterr().out
    assert "backend.allreduce_dispatch" in out
    assert "ep:0" in out


def test_diff_reads_one_side_from_stdin(monkeypatch, capsys, tmp_path):
    raw = (FIX / "vllm-0.29-tp4-ep-h200-allreduce.log").read_bytes()
    other = tmp_path / "cand.log"
    other.write_text("INFO [cuda_communicator.py:245] Using ['PYNCCL'] all-reduce backends "
                     "(in dispatch order) for group 'tp:0' out of potential backends: ['PYNCCL']\n",
                     encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"))
    assert servediff.main(["diff", "-", str(other), "--markdown"]) == 1
    out = capsys.readouterr().out
    assert "stdin" in out
    assert "ep:0" in out
    # stdin can only be consumed once, so both sides cannot be '-'
    assert servediff.main(["diff", "-", "-"]) == 2


def test_audit_survives_a_single_byte_console_code_page():
    # A Windows console defaults to a single-byte code page, and the report echoes a
    # box-drawing banner back. That raised UnicodeEncodeError and exited 1 -- and 1 means
    # "the inputs differ", so a CI gate read a console limitation as a config change.
    import os
    log = FIX / "vllm-0.33.0-real-rtx4060.log"
    assert "\u2588" in log.read_text(encoding="utf-8"), "the fixture must carry the banner"
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    proc = subprocess.run([sys.executable, "-m", "servediff", "audit", str(log)],
                          capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "UnicodeEncodeError" not in (proc.stdout + proc.stderr)
    assert "Traceback" not in (proc.stdout + proc.stderr)


def test_show_json_survives_a_single_byte_console_code_page():
    # `show --json` writes the payload in one print(); when the encoding failed nothing
    # was written, and `--against` then read an empty facts file.
    import os
    log = FIX / "vllm-0.33.0-real-rtx4060.log"
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    proc = subprocess.run([sys.executable, "-m", "servediff", "show", str(log), "--json"],
                          capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr[-400:]
    payload = json.loads(proc.stdout)
    assert payload["engine.version"]["value"]


def test_series_keeps_the_middle_column(capsys):
    # vllm#53472 carried three runs of the same config. Two pairwise diffs lose the
    # middle column; one series table keeps every version side by side.
    logs = [FIX / "vllm-53472-v0.23.log", FIX / "vllm-53472-v0.26.log",
            FIX / "vllm-53472-v0.26-flag.log"]
    parsed = [servediff.parse(path.read_text(encoding="utf-8")) for path in logs]
    rows = {row["key"]: row for row in servediff.series(parsed)}
    assert len(next(iter(rows.values()))["values"]) == 3
    assert any(row["status"] == "changed" for row in rows.values())
    assert servediff.main(["series", *map(str, logs)]) == 1
    out = capsys.readouterr().out
    assert "differ across 3 logs" in out


    # two logs belong to `diff`
    assert servediff.main(["series", str(logs[0]), str(logs[1])]) == 2
    # --all adds the identical rows back, and the markdown keeps one column per log
    assert servediff.main(["series", *map(str, logs), "--all", "--markdown"]) == 1
    out = capsys.readouterr().out
    assert "identical" in out
    assert out.splitlines()[0].count("|") == 5


def test_env_to_a_path_that_cannot_be_written_is_exit_2(tmp_path, capsys, monkeypatch):
    # The README promises `2 snapshot failed`. A typo in -o used to end in
    # FileNotFoundError with exit 1, which reads as "one difference" to a calling script.
    monkeypatch.setattr(servediff.envsnap, "snapshot",
                        lambda *a, **k: {"schema": "servediff-env/1", "packages": {}, "env": {}})
    missing = tmp_path / "nodir" / "snap.json"
    assert servediff.main(["env", "-o", str(missing)]) == 2
    err = capsys.readouterr().err
    assert "cannot write" in err
    assert "Traceback" not in err


def test_capture_to_a_path_that_cannot_be_written_is_exit_2(tmp_path, capsys, monkeypatch):
    # Nothing has been launched when the paths cannot be opened, so this is a usage error
    # and never the server's exit code.
    monkeypatch.setattr(servediff.envsnap, "snapshot",
                        lambda *a, **k: {"schema": "servediff-env/1", "packages": {}, "env": {}})
    missing = tmp_path / "nodir" / "run.log"
    assert servediff.main(["capture", "-o", str(missing), "--", "fake-server"]) == 2
    err = capsys.readouterr().err
    assert "cannot write" in err
    assert "Traceback" not in err


def test_series_text_keeps_values_distinct_after_truncation():
    """Two values that differ only at the end must not render as the same text; the row is
    labelled "changed", so identical-looking cells would read as a bug in the report."""
    rows = [{"key": "backend.moe_prepare_finalize",
             "values": ["MoEPrepareAndFinalizeNoDPEPMonolithic",
                        "MoEPrepareAndFinalizeNoDPEPModular"]}]
    text = servediff._render_text_series(["a.log", "b.log"], rows, width=30)
    assert "nolithic" in text and "Modular" in text
    assert "MoEPrepareAndFinalizeNoDPEP..." not in text


def test_truncated_dispatch_line_still_yields_the_chain(tmp_path):
    # Log sinks cut long lines, and this one carries a seven-item potential list.
    # The truncated form must still yield the chain, while the same sentence quoted
    # as prose in an issue body must not.
    log = tmp_path / "trunc.log"
    log.write_text(
        "INFO [cuda_communicator.py:269] Using ['FLASHINFER', 'CUSTOM', 'PYNCCL'] all-reduce "
        "backends (in dispatch order) for group 'tp:0' ...\n"
        "Startup line `Using ['FLASHINFER'] all-reduce backends (in dispatch order) for group "
        "'tp:0'` on 2x H100 SXM, TP=2\n",
        encoding="utf-8")
    f = servediff.parse(log.read_text(encoding="utf-8"))["facts"]
    assert f["backend.allreduce_dispatch"].values == ["tp:0: ['FLASHINFER', 'CUSTOM', 'PYNCCL']"]
    assert "backend.allreduce_potential" not in f


def test_single_rank_log_explains_the_missing_allreduce_path(tmp_path):
    # A TP=1 log has no all-reduce group at all, so nothing is printed. Saying so
    # is what keeps "absent" from being read as "unchanged".
    log = tmp_path / "tp1.log"
    log.write_text(
        "INFO 10-28 11:28:09 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='m'\n"
        "INFO 10-28 11:28:12 [parallel_state.py:1827] world_size=1 rank=0 local_rank=0\n",
        encoding="utf-8")
    notes = servediff.absence_notes(servediff.parse(log.read_text(encoding="utf-8")))
    assert len(notes) == 1
    key, text = notes[0]
    assert key == "backend.allreduce_dispatch"
    assert "world_size=1" in text and "TP>1" in text


def test_flashinfer_absence_is_explained_only_from_029(tmp_path):
    # v0.29.0 moved the auto-selection line to DEBUG; before that its absence
    # really did mean the path was not taken, so the note must not fire there.
    chain = ("INFO [cuda_communicator.py:1] Using ['FLASHINFER', 'PYNCCL'] all-reduce backends "
             "(in dispatch order) for group 'tp:0' out of potential backends: "
             "['FLASHINFER', 'PYNCCL']\n")
    for version, expected in (("0.26.0", False), ("0.29.0", True)):
        log = tmp_path / f"v{version}.log"
        log.write_text(
            f"INFO [core.py:122] Initializing a V1 LLM engine (v{version}) with config: model='m'\n" + chain,
            encoding="utf-8")
        notes = servediff.absence_notes(servediff.parse(log.read_text(encoding="utf-8")))
        assert any("DEBUG" in text for _, text in notes) is expected


def test_markdown_reports_absence_notes(tmp_path, capsys):
    body = ("INFO [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='m'\n"
            "INFO [parallel_state.py:1827] world_size=1 rank=0 local_rank=0\n")
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(body, encoding="utf-8")
    b.write_text(body, encoding="utf-8")
    assert servediff.main(["diff", str(a), str(b), "--markdown"]) == 0
    out = capsys.readouterr().out
    assert "ℹ️" in out and "world_size=1" in out


def test_only_drops_the_notes_about_facts_it_excluded(tmp_path, capsys):
    # `--only attention` narrowed the comparison to one row, and the note about the
    # all-reduce path it excluded was still printed above the table: three paragraphs
    # explaining a fact the reader had just said they were not comparing.
    body = ("INFO [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='m'\n"
            "INFO [parallel_state.py:1827] world_size=1 rank=0 local_rank=0\n"
            "INFO [cuda.py:538] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN']\n")
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(body, encoding="utf-8")
    b.write_text(body.replace("FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN']",
                              "TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN']"),
                 encoding="utf-8")
    assert servediff.main(["diff", str(a), str(b), "--only", "attention"]) == 1
    out = capsys.readouterr().out
    assert "world_size=1" not in out, out  # the all-reduce note is about an excluded fact
    assert "backend.attention" in out and "changed" in out

    # Naming the all-reduce path again brings its note back, unchanged.
    assert servediff.main(["diff", str(a), str(b), "--only", "backend."]) == 1
    assert "world_size=1" in capsys.readouterr().out


def test_a_facts_file_gates_one_log_without_shipping_the_other(tmp_path, capsys):
    # A CI job has today's log and no yesterday's: keeping the old log in the repo is
    # what nobody does. `show --json` records the facts of a good run, and --against must
    # produce the rows a two-log diff would, so the gate works on one log at a time.
    good, bad = tmp_path / "good.log", tmp_path / "bad.log"
    good.write_text(FALLBACK_STARTUP, encoding="utf-8")
    bad.write_text(FAILED_STARTUP, encoding="utf-8")
    facts = tmp_path / "facts.json"
    assert servediff.main(["show", str(good), "--json"]) == 0
    facts.write_text(capsys.readouterr().out, encoding="utf-8")

    assert servediff.main(["diff", str(good), "--against", str(facts)]) == 0
    out = capsys.readouterr().out
    assert "0 difference(s)" in out
    assert "recorded from an earlier log, not a log" in out

    # A different run must fail the gate, and the reason must be visible: the facts file
    # cannot explain why a fact is missing, only that it never recorded it.
    assert servediff.main(["diff", str(bad), "--against", str(facts)]) == 1
    out = capsys.readouterr().out
    assert "only_baseline" in out


def test_a_facts_file_round_trips_a_multi_valued_fact(tmp_path, capsys):
    # `value` collapses to a scalar or a list depending on how many values a fact had,
    # which is exactly the ambiguity that would silently turn two conflicting backends
    # into one. --against reads `values`, so the round trip has to be exact.
    log = tmp_path / "two.log"
    log.write_text("Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN']\n"
                   "Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN']\n",
                   encoding="utf-8")
    facts = tmp_path / "facts.json"
    assert servediff.main(["show", str(log), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend.attention"]["values"] == ["FLASH_ATTN", "TRITON_ATTN"]
    facts.write_text(json.dumps(payload), encoding="utf-8")
    assert servediff.main(["diff", str(log), "--against", str(facts)]) == 0


def test_a_file_that_is_not_a_facts_file_is_an_input_error(tmp_path, capsys):
    # Exit 2, not a traceback and not a difference: a broken baseline makes the gate
    # wrong, and "wrong" is not the same finding as "the server changed".
    log = tmp_path / "a.log"
    log.write_text(FALLBACK_STARTUP, encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text('{"x": 1}', encoding="utf-8")
    assert servediff.main(["diff", str(log), "--against", str(bad)]) == 2
    assert "not a facts file" in capsys.readouterr().err

    assert servediff.main(["diff", str(log), "--against", str(tmp_path / "nope.json")]) == 2
    assert "No such file" in capsys.readouterr().err

    assert servediff.main(["diff", str(log), str(log), "--against", str(bad)]) == 2
    assert "drop the second log" in capsys.readouterr().err


NARROW_A = ("INFO [core.py:122] Initializing a V1 LLM engine (v0.29.0) with config: model='m'\n"
            "Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN']\n"
            "[pip3] numpy==2.4.6\n")
NARROW_B = ("INFO [core.py:122] Initializing a V1 LLM engine (v0.30.0) with config: model='m'\n"
            "Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN']\n"
            "[pip3] numpy==2.2.6\n")


def test_only_one_prefix_watches_one_decision(tmp_path, capsys):
    # A gate that goes red whenever a package pin moves is a gate that stays red, so the
    # CI wants "the attention backend did not change" on its own. --only must compare just
    # that row and must say how much it left out, or a narrowed pass reads as a clean one.
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(NARROW_A, encoding="utf-8")
    b.write_text(NARROW_B, encoding="utf-8")
    assert servediff.main(["diff", str(a), str(b), "--only", "backend.attention"]) == 1
    out = capsys.readouterr().out
    assert "FLASH_ATTN" in out and "TRITON_ATTN" in out
    assert "engine.version" not in out, out
    assert "env.pip.numpy" not in out, out
    assert "only 1 fact(s) matched --only backend.attention" in out
    assert "the exit status covers only these" in out


def test_only_takes_several_prefixes(tmp_path, capsys):
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(NARROW_A, encoding="utf-8")
    b.write_text(NARROW_B, encoding="utf-8")
    assert servediff.main(["diff", str(a), str(b), "--only", "engine.", "--only", "env.pip.numpy"]) == 1
    out = capsys.readouterr().out
    assert "engine.version" in out and "env.pip.numpy" in out
    assert "backend.attention" not in out, out


def test_a_prefix_that_matches_nothing_is_exit_2(tmp_path, capsys):
    # `--only cuda_graph` on two logs that never printed it would otherwise be a green
    # gate that watches nothing: the same failure as an unreadable baseline.
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(NARROW_A, encoding="utf-8")
    b.write_text(NARROW_B, encoding="utf-8")
    assert servediff.main(["diff", str(a), str(b), "--only", "cuda_graph"]) == 2
    err = capsys.readouterr().err
    assert "matched no fact" in err and "a gate on nothing is not a pass" in err


ALLREDUCE_A = (
    "INFO [parallel_state.py:961] world_size=4 rank=0 local_rank=0\n"
    "INFO [cuda_communicator.py:112] Using ['PYNCCL'] all-reduce backends "
    "(in dispatch order) for group 'tp:0' out of potential backends: ['PYNCCL']\n"
)
ALLREDUCE_B = (
    "INFO [parallel_state.py:961] world_size=4 rank=0 local_rank=0\n"
    "INFO [cuda_communicator.py:112] Using ['FLASHINFER', 'PYNCCL'] all-reduce backends "
    "(in dispatch order) for group 'tp:0' out of potential backends: ['PYNCCL', 'FLASHINFER']\n"
)


def test_the_word_from_the_log_finds_the_fact_it_means(tmp_path, capsys):
    # The question an outside reader actually asks — "did the all-reduce chain change?" —
    # is spelled the way the log spells it (`all-reduce`), never the way the fact name is
    # spelled (`backend.allreduce_dispatch`). A prefix-only match answered that question
    # with exit 2, i.e. "matched nothing", which reads as a broken gate rather than a miss.
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(ALLREDUCE_A, encoding="utf-8")
    b.write_text(ALLREDUCE_B, encoding="utf-8")
    for spelling in ("all_reduce", "all-reduce", "allreduce"):
        assert servediff.main(["diff", str(a), str(b), "--only", spelling]) == 1, spelling
        out = capsys.readouterr().out
        assert "backend.allreduce_dispatch" in out, spelling
        assert "changed" in out, spelling
        assert "parallel.world_size" not in out, spelling

    # The boundary stays where it was: --only names facts, never values. A word that only
    # appears in a value (`FLASHINFER`) still matches nothing, so a gate cannot accidentally
    # pin itself to whatever a version printed that week.
    assert servediff.main(["diff", str(a), str(b), "--only", "FLASHINFER"]) == 2
    assert "a gate on nothing is not a pass" in capsys.readouterr().err

    # A pattern that still matches nothing keeps the loud exit: the gate is broken, not clean.
    assert servediff.main(["diff", str(a), str(b), "--only", "cu_graph_magic"]) == 2
    assert "a gate on nothing is not a pass" in capsys.readouterr().err


def test_only_is_rejected_on_env_snapshots(tmp_path, capsys):
    snap = {"schema": "servediff-env/1", "python": "3.12.3", "gpus": [], "packages": {"vllm": "0.30.0"}, "env": {}}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(snap), encoding="utf-8")
    b.write_text(json.dumps({**snap, "packages": {"vllm": "0.31.0"}}), encoding="utf-8")
    assert servediff.main(["diff", str(a), str(b), "--only", "pkg.vllm"]) == 2
    assert "env snapshots are compared whole" in capsys.readouterr().err


def test_a_typo_names_the_command_not_the_module(capsys):
    # A first-timer mistypes the subcommand, and the fix-it line said `usage: __main__.py
    # ...` — a file they never created. `servediff` is what they typed and what the console
    # script is called; both invocations must say the same thing.
    with __import__("pytest").raises(SystemExit) as exc:
        servediff.main(["dif"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: servediff "), err[:80]
    assert "__main__.py" not in err


def test_the_other_phrasing_of_the_kv_cache_dtype_is_read(tmp_path):
    # Audit over a stranger's log (vllm#66423, a segfault report we never shipped a
    # fixture for) flagged `Using fp8 data type to store kv cache.` as a shape worth a
    # look: the rule that produced kv_cache.dtype wanted SGLang's wording. Two engines
    # phrase the same decision differently, and only one phrasing was covered.
    log = tmp_path / "a.log"
    log.write_text(
        "INFO [core.py:123] Initializing a V1 LLM engine (v0.31.0) with config: model='m'\n"
        "INFO [cache.py:245] Using fp8 data type to store kv cache.\n",
        encoding="utf-8")
    facts = servediff.parse(log.read_text(encoding="utf-8"))["facts"]
    assert facts["kv_cache.dtype"].values == ["fp8"]

    # The SGLang wording still lands in the same fact, so a cross-engine comparison of
    # the same decision does not split into two rows.
    other = tmp_path / "b.log"
    other.write_text("INFO  KV Cache is allocated. dtype: fp8, size: 1.5 GB\n", encoding="utf-8")
    assert servediff.parse(other.read_text(encoding="utf-8"))["facts"]["kv_cache.dtype"].values == ["fp8"]


CORPUS_BASE = ("INFO [core.py:122] Initializing a V1 LLM engine (v0.29.0) with config: model='m'\n"
               "Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN']\n"
               "[pip3] numpy==2.4.6\n")
CORPUS_SAME = CORPUS_BASE.replace("numpy==2.4.6", "numpy==2.4.7")
CORPUS_BAD = ("INFO [core.py:122] Initializing a V1 LLM engine (v0.31.0) with config: model='m'\n"
              "Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN']\n"
              "[pip3] numpy==2.4.6\n")


def _corpus(tmp_path, capsys):
    """A CI matrix: one log per configuration, plus the facts recorded from the accepted one."""
    d = tmp_path / "runs"
    d.mkdir()
    (d / "base.log").write_text(CORPUS_BASE, encoding="utf-8")
    (d / "same.log").write_text(CORPUS_SAME, encoding="utf-8")
    (d / "bad.log").write_text(CORPUS_BAD, encoding="utf-8")
    assert servediff.main(["show", str(d / "base.log"), "--json"]) == 0
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(capsys.readouterr().out, encoding="utf-8")
    return d, facts_path


def test_a_directory_gates_every_log_and_reports_one_exit_code(tmp_path, capsys):
    # Six matrix rows should not need six diffs: one command, one line per log, one exit
    # code, and the failing row named without reading a table.
    d, facts_path = _corpus(tmp_path, capsys)
    assert servediff.main(["diff", str(d), "--against", str(facts_path)]) == 1
    out = capsys.readouterr().out
    # The candidate list rides on the selection line ("out of potential backends: [...]"),
    # so a matrix row that switches backend changes two facts, not one: the chosen kernel
    # and the set the build was willing to choose from.
    assert "bad.log" in out and "backend.attention, backend.candidates, engine.version" in out
    assert "base.log" in out and "0 difference(s); 4 fact(s) identical" in out
    assert "3 log(s): 2 differ from" in out and "1 unchanged" in out


def test_a_directory_respects_only_so_one_row_fails(tmp_path, capsys):
    d, facts_path = _corpus(tmp_path, capsys)
    assert servediff.main(["diff", str(d), "--against", str(facts_path), "--only", "backend."]) == 1
    out = capsys.readouterr().out
    assert "1 differ from" in out and "2 unchanged" in out
    assert "[backend.attention, backend.candidates]" in out, out


def test_a_directory_without_logs_is_an_input_error(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    base = tmp_path / "base.log"
    base.write_text(CORPUS_BASE, encoding="utf-8")
    assert servediff.main(["show", str(base), "--json"]) == 0
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(capsys.readouterr().out, encoding="utf-8")
    assert servediff.main(["diff", str(empty), "--against", str(facts_path)]) == 2
    assert "no *.log inside" in capsys.readouterr().err


def test_a_blank_artifact_in_the_directory_fails_the_gate(tmp_path, capsys):
    # A truncated matrix artifact is a broken gate input, not a run that happened to match.
    d, facts_path = _corpus(tmp_path, capsys)
    (d / "blank.log").write_text("", encoding="utf-8")
    assert servediff.main(["diff", str(d), "--against", str(facts_path)]) == 2
    assert "no vLLM or SGLang configuration lines recognised" in capsys.readouterr().err


def test_one_log_without_against_is_an_input_error(tmp_path, capsys):
    # A newcomer's first move is "diff the one log I have". Say so in a sentence, and never
    # let it come out as a traceback on the tool's main command.
    log = tmp_path / "startup.log"
    log.write_text(CORPUS_BASE, encoding="utf-8")
    assert servediff.main(["diff", str(log)]) == 2
    err = capsys.readouterr().err
    assert "one log is not a comparison" in err
    assert "--against" in err
    assert "Traceback" not in err


WAIT = ("No available shared memory broadcast block found in 60 seconds."
        " This typically happens when some processes are hanging or doing some"
        " time-consuming work (e.g. compilation, weight/kv cache quantization).")


def _write_log(tmp_path, lines, name="repeats.log"):
    log = tmp_path / name
    log.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return log


def test_repeats_reads_the_clock_out_of_a_bracketed_sglang_timestamp(tmp_path, capsys):
    # SGLang prints `[2026-10-06 00:15:19] ...`. Stripping the bracketed preamble used to
    # take the only clock on the line with it, so every SGLang log reported "no timestamps"
    # and the whole of `repeats` was blind on that side of the coverage.
    log = _write_log(tmp_path, [
        f"[2026-10-06 00:15:{minute:02d}] NIXL KVManager initialized with backend: UCX"
        for minute in (19, 20, 21)])
    assert servediff.main(["repeats", str(log)]) == 0
    out = capsys.readouterr().out
    assert "untimed" not in out
    assert "3x" in out
    assert "00:15:19 -> 00:15:21" in out


def test_repeats_reads_a_fixed_gap_as_one_wait(tmp_path, capsys):
    # Sixteen copies of vLLM's ring-buffer warning are one wait re-logged every
    # VLLM_RINGBUFFER_WARNING_INTERVAL, not sixteen stalls. The gap is the only signal that
    # separates the two, and counting lines throws it away.
    log = _write_log(tmp_path, [
        f"(EngineCore pid=260) INFO 08-31 11:{minute:02d}:32 shm_broadcast.py:801] {WAIT}"
        for minute in (34, 35, 36, 37, 38)])
    assert servediff.main(["repeats", str(log)]) == 0
    out = capsys.readouterr().out
    assert "5x" in out
    assert "regular" in out
    assert "span 240 s" in out
    assert "60.000 s median" in out
    assert "one wait being re-logged" in out


def test_repeats_reads_a_tight_burst_as_retries(tmp_path, capsys):
    # An allocator retrying an allocation prints the same line every few milliseconds:
    # regular it is not, and the median gap says so.
    log = _write_log(tmp_path, [
        f"(Worker_TP0_EP0 pid=356) WARNING 08-31 11:33:26.{ms:03d} cuda.py:1] "
        "memory allocation failed with OOM while trying to allocate 20971520 bytes"
        for ms in (100, 112, 124, 136)])
    assert servediff.main(["repeats", str(log)]) == 0
    out = capsys.readouterr().out
    assert "4x" in out
    assert "burst" in out
    assert "0.012 s median" in out
    assert "one wait being re-logged" not in out


def test_repeats_splits_a_warning_spliced_onto_a_progress_bar(tmp_path, capsys):
    # A tqdm bar rewrites one terminal row with \r and carries no newline, so a captured log
    # puts the next process's line on the same physical line. Left unsplit, the warning hides
    # behind the bar's digits and never groups with its own copies.
    spliced = ("Capturing CUDA graphs (PIECEWISE):  100%|##########| 51/51 [00:01<00:00, 3.2s/it]"
               f"(EngineCore pid=260) INFO 08-31 11:37:30 shm_broadcast.py:801] {WAIT}")
    log = _write_log(tmp_path, [
        spliced,
        f"(EngineCore pid=260) INFO 08-31 11:38:30 shm_broadcast.py:801] {WAIT}"])
    assert servediff.main(["repeats", str(log)]) == 0
    out = capsys.readouterr().out
    assert "2x" in out
    assert "span 60" in out
    # the reported sample is the warning itself, not the progress bar it was glued to
    assert "########" not in out.split("No available")[0].splitlines()[-1]


def test_repeats_says_untimed_when_a_line_carries_no_clock(tmp_path, capsys):
    # A merged multi-process log keeps lines from files that never printed a timestamp;
    # the count is still useful, the gap is not invented.
    log = _write_log(tmp_path, [
        "[transformers] Unrecognized keys in `rope_parameters`",
        "[transformers] Unrecognized keys in `rope_parameters`"])
    assert servediff.main(["repeats", str(log)]) == 0
    out = capsys.readouterr().out
    assert "2x" in out
    assert "untimed" in out
    assert "gap unknown" in out


def test_repeats_json_carries_the_span_and_the_median(tmp_path, capsys):
    log = _write_log(tmp_path, [
        f"(EngineCore pid=260) INFO 08-31 11:{minute:02d}:00 core.py:1] {WAIT}"
        for minute in (10, 11, 12)])
    assert servediff.main(["repeats", str(log), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    group = payload[0]["repeated"][0]
    assert group["count"] == 3
    assert group["regular"] is True
    assert group["span_s"] == 120.0
    assert group["median_gap_s"] == 60.0
    assert group["first"] == 11 * 3600 + 10 * 60  # the log quotes 11:10:00


def test_show_evidence_prints_the_line_a_fact_came_from(capsys):
    # `show` prints our reading and a line number. To disagree with the reading you need the
    # server's own text, and finding line 6 of a 100 MB file by hand is where that stops.
    log = FIX / "vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log"
    assert servediff.main(["show", str(log)]) == 0
    plain = capsys.readouterr().out
    assert "line 6:" not in plain
    assert servediff.main(["show", str(log), "--evidence"]) == 0
    out = capsys.readouterr().out
    assert "[backends.py:1148] Dynamo bytecode transform time: 9.35 s" in out
    assert "line 6: " in out


def test_show_markdown_is_a_pasteable_one_log_report(capsys):
    # A `diff` report needs two logs, so a reader with one log and a question had nothing
    # to paste but fixed-width terminal output, which reads in an issue as a screenshot of
    # somebody's terminal. The one-log report keeps the two columns that survive a reflow.
    log = FIX / "sglang-2026-10/sglang-42684.log"
    assert servediff.main(["show", str(log), "--markdown"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# servediff report: one log")
    assert "| Fact | Value | Line |" in out
    assert "| `engine.version` | 0.5.20 |" in out
    # `_fmt` trims at 60 characters for a terminal column; a value cut mid-word in a
    # report pasted into an issue is a fact the reader cannot use.
    row = [line for line in out.splitlines() if line.startswith("| `cache.radix`")][0]
    assert "..." not in row
    assert "<details>" not in out  # evidence stays opt-in
    assert servediff.main(["show", str(log), "--markdown", "--evidence"]) == 0
    assert "<summary>Evidence</summary>" in capsys.readouterr().out
    assert servediff.main(["show", str(log), "--markdown", "--json"]) == 2


def test_bare_servediff_shows_where_to_start(capsys):
    # The first command a person runs after `pip install` is `servediff`. It used to answer
    # an argparse error, because `command` is required and they had no log yet.
    assert servediff.main([]) == 0
    out = capsys.readouterr().out
    assert "servediff demo" in out
    assert "servediff show server.log" in out
    assert "apex-mochen.github.io/servediff" in out


def test_the_candidate_backend_list_is_read_from_the_selection_line():
    # The rule was written for a bare `backends: [...]` line, which vLLM does not print:
    # the list rides on the selection line itself. So every real log had the chosen backend
    # as a fact and the set it was chosen from invisible -- and a build that stopped
    # offering FLASHINFER diffed as "FLASH_ATTN, same as before".
    log = FIX / "vllm-0.33.0-real-rtx4060.log"
    facts = servediff.parse(log.read_text(encoding="utf-8", errors="replace"))["facts"]
    assert facts["backend.candidates"].value == (
        "['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']")
    assert facts["backend.candidates"].evidence[0][0] == 27
    assert "out of potential backends" in facts["backend.candidates"].evidence[0][1]


def test_two_values_that_differ_past_the_column_are_not_printed_identically(tmp_path, capsys):
    # A 40-column cell is what keeps a diff readable, and it also hides a change that
    # starts at column 41: the row then reads `changed` above two identical strings. The
    # tool exists to show differences a person cannot eyeball, so this one has to be
    # visible: the cells widen just past where they diverge, not to full width.
    line = ("Using FLASH_ATTN attention backend out of potential backends: "
            "['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']\n")
    a = tmp_path / "a.log"
    b = tmp_path / "b.log"
    a.write_text("INFO [core.py:122] Initializing a V1 LLM engine (v0.34.0) with config: model='m'\n" + line,
                 encoding="utf-8")
    b.write_text("INFO [core.py:122] Initializing a V1 LLM engine (v0.34.0) with config: model='m'\n"
                 + line.replace("'FLEX_ATTENTION']", "'FLEX_ATTENTION', 'ZZZ']"), encoding="utf-8")
    assert servediff.main(["diff", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    row = [l for l in out.splitlines() if l.startswith("backend.candidates")][0]
    assert row.count("changed") == 1
    assert "'FLEX_ATTENTION']" in row and "'FLEX_ATTENTION', 'ZZZ']" in row, row


def _corpus_keys():
    text = (FIX / "expected_fact_keys.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def test_no_fact_value_joins_two_numbers_with_a_bare_comma():
    # `compile.range` is the pair `(1, 2048)`, and joining the two captures with a comma
    # alone rendered `1,2048` -- one malformed number with a misplaced thousands separator,
    # in every table, for several releases, because a test had asserted the broken string.
    # This walks the whole corpus instead: a thousands separator has exactly three digits
    # after the comma, so anything else is two numbers that lost their separator.
    bad = []
    for path in sorted(FIX.rglob("*.log")):
        facts = servediff.parse(path.read_text(encoding="utf-8", errors="replace"))["facts"]
        for key, fact in facts.items():
            for value in fact.values:
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                for match in re.finditer(r"\d{1,3},(\d+)", text):
                    if len(match.group(1)) != 3:
                        bad.append((path.name, key, text[:80]))
    assert not bad, bad[:5]


def test_diff_why_explains_the_changed_keys_and_groups_them(capsys):
    # A diff row states a change and stops. --why answers "so what does this count" in the
    # same command, and groups keys by the entry that explains them: one version bump that
    # moves six compile.* keys is one explanation, not six.
    a = FIX / "vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log"
    b = FIX / "sglang-2026-10/sglang-42684.log"
    assert servediff.main(["diff", str(a), str(b), "--why"]) == 1
    out = capsys.readouterr().out
    assert "what these changed keys measure" in out
    assert "does not tell you" in out
    assert "compile  (" in out  # six keys, one heading
    # A key that did not change is not explained: the section is about this diff.
    tail = out.split("what these changed keys measure", 1)[1]
    assert "kv_cache.max_concurrency" not in tail


def test_diff_why_json_groups_without_changing_the_rows(capsys):
    a = FIX / "vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log"
    b = FIX / "sglang-2026-10/sglang-42684.log"
    assert servediff.main(["diff", str(a), str(b), "--why", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert "why" in data and len(data["why"]) <= 6
    assert data["why_omitted"] > 0  # 46 changed keys cannot all be explained in one screen
    assert all("trap" in g and "keys" in g for g in data["why"].values())
    assert all("rows" not in g for g in data["why"].values())
    assert set(data["rows"][0]) >= {"key", "baseline", "candidate", "status"}


def test_diff_why_markdown_carries_the_section(capsys):
    # The markdown report is the thing pasted into an issue, so the explanation has to be
    # inside the pasted text -- on the terminal it lands on the wrong side of the copy.
    a = FIX / "vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log"
    b = FIX / "sglang-2026-10/sglang-42684.log"
    assert servediff.main(["diff", str(a), str(b), "--why", "--markdown"]) == 1
    out = capsys.readouterr().out
    assert "## What the changed keys mean" in out
    assert "*Does not tell you:*" in out
    assert "not explained here" in out  # the omitted count is stated, not dropped


def test_markdown_without_why_has_no_explanation_section(capsys):
    a = FIX / "vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log"
    b = FIX / "sglang-2026-10/sglang-42684.log"
    assert servediff.main(["diff", str(a), str(b), "--markdown"]) == 1
    assert "## What the changed keys mean" not in capsys.readouterr().out


def test_series_why_answers_the_same_question(capsys):
    # Three-way comparisons raise the same "what does this count" question as diff; the
    # flag existing on one command and not the other is the kind of gap a reader hits.
    logs = [str(FIX / "vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log"),
            str(FIX / "sglang-2026-10/sglang-42684.log"),
            str(FIX / "vllm-forum-2026-06/vllm-cpu-2x-epyc-9655.log")]
    assert servediff.main(["series", *logs]) == 1
    assert "does not tell you" not in capsys.readouterr().out
    assert servediff.main(["series", *logs, "--why"]) == 1
    out = capsys.readouterr().out
    assert "what these changed keys measure" in out and "does not tell you" in out
    assert servediff.main(["series", *logs, "--why", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["why_omitted"] > 0 and len(data["why"]) <= 6


def test_diff_without_why_says_nothing_extra(capsys):
    a = FIX / "vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log"
    b = FIX / "sglang-2026-10/sglang-42684.log"
    assert servediff.main(["diff", str(a), str(b)]) == 1
    assert "what these changed keys measure" not in capsys.readouterr().out


def test_every_fact_key_can_be_explained(capsys):
    # The meaning of a key used to live in the comment above its regex -- in the source, where
    # a reader of `show` output never looks. The glossary is the promise that every key the
    # tool prints can be read; this gate keeps the promise as keys are added.
    keys = _corpus_keys()
    assert len(keys) > 250
    missing = []
    for key in keys:
        capsys.readouterr()
        if servediff.main(["explain", key]) != 0:
            missing.append(key)
    assert missing == [], f"no explanation for: {missing}"


def test_the_vocabulary_shipped_in_the_package_matches_the_corpus():
    # `explain` needs the key list to tell a key covered by its family from a typo; if the
    # list drifts, a new key starts answering with "no entry" on a real log.
    from servediff.glossary import KNOWN_KEYS
    assert set(KNOWN_KEYS) == set(_corpus_keys())


def test_glossary_only_points_at_keys_the_corpus_produces():
    # A trap that names `kv_cache.requested_gib` is only useful while that key exists; a
    # renamed key turns the cross-reference into a dead pointer that still reads correct.
    from servediff.glossary import GLOSSARY, KNOWN_KEYS
    fams = {k.split(".")[0] for k in KNOWN_KEYS}
    prose = {"e.g", "torch.compile"}  # names of things, not fact keys
    bad = []
    for name, gloss in GLOSSARY.items():
        text = f"{gloss.what} {gloss.trap}"
        for token in re.findall(r"[a-z_]+(?:\.?[a-z_0-9]+)+", text):
            if "." not in token or token in prose or token in KNOWN_KEYS or token in GLOSSARY:
                continue
            if token.split(".")[0] in fams:  # a family prefix such as env.pip
                continue
            bad.append((name, token))
    assert bad == []


def test_explain_calls_a_typo_a_typo(capsys):
    # A dotted key that no log produces used to answer with its family gloss and exit 0,
    # which reads exactly like documentation. It is a typo and must say so.
    assert servediff.main(["explain", "kv_cache.tokenz"]) == 2
    err = capsys.readouterr().err
    assert "no entry for 'kv_cache.tokenz'" in err
    assert "kv_cache.tokens" in err


def test_explain_a_known_key_prints_the_family_and_the_trap(capsys):
    assert servediff.main(["explain", "kv_cache.max_concurrency"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("kv_cache.max_concurrency")
    assert "sequences" in out
    assert "family: kv_cache" not in out  # an exact entry does not repeat its family
    assert "does not tell you" in out


def test_explain_without_a_key_lists_the_families(capsys):
    assert servediff.main(["explain"]) == 0
    out = capsys.readouterr().out
    assert "kv_cache" in out and "env" in out and "startup" in out


def test_explain_json_is_parseable(capsys):
    assert servediff.main(["explain", "--json", "torch.oom_holder_gib"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["family"] == "torch"
    assert "94.41" in data["trap"]


def test_show_only_says_nothing_when_the_log_had_nothing_to_read(capsys):
    # `show` exits 2 with "no configuration lines recognised". That sentence is a promise:
    # on a log that does carry facts it must never be printed, or the tool has quietly
    # stopped covering a real server's output. This walks the whole corpus rather than the
    # hand-picked logs the other tests use.
    for path in sorted(FIX.rglob("*.log")):
        facts = servediff.parse(path.read_text(encoding="utf-8", errors="replace"))["facts"]
        assert servediff.main(["show", str(path)]) == (0 if facts else 2), path
        capsys.readouterr()


def test_repeats_survives_a_repeat_whose_lines_share_one_clock():
    # vllm-project/vllm forum, `ray-4-node-random-fail`: five copies of the same traceback
    # frame, all stamped 16:38:14. `count >= 2` with a single distinct clock reached a
    # `None` median and the formatter raised TypeError — `repeats` died on the log instead
    # of saying the gap is unreadable.
    for name in ("vllm-forum-2026-04/ray-4-node-random-fail.log",
                 "vllm-forum-2025-07/qwen-vram-enough-but-oom.log"):
        assert servediff.main(["repeats", str(FIX / name)]) == 0


def test_repeats_min_hides_a_shape_below_the_bar(tmp_path, capsys):
    log = _write_log(tmp_path, [
        f"(EngineCore pid=260) INFO 08-31 11:{minute:02d}:00 core.py:1] {WAIT}"
        for minute in (10, 11)])
    assert servediff.main(["repeats", str(log), "--min", "3"]) == 0
    out = capsys.readouterr().out
    assert "nothing repeats 3 or more times." in out


def test_repeats_reads_a_log_from_stdin(monkeypatch, capsys):
    text = "".join(
        f"(EngineCore pid=260) INFO 08-31 11:{minute:02d}:00 core.py:1] {WAIT}\n"
        for minute in (10, 11, 12))
    raw = text.encode("utf-8")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"))
    assert servediff.main(["repeats", "-"]) == 0
    out = capsys.readouterr().out
    assert "3x" in out
    assert "regular" in out

def test_no_fact_key_disappears_from_the_corpus():
    """A rule that stops matching fails nothing on its own — the table just gets shorter.

    Removing the `worker.mp_method` rule by accident while inserting a new one left the
    suite green on every test that did not happen to assert that key. This compares the
    keys produced by the whole bundled corpus against a checked-in inventory, so any loss
    or addition has to be acknowledged (`python tools/fact_keys.py --write`). Scope: keys
    that only exist in an inline test string (like `worker.mp_method`) are guarded by the
    test that writes that string, not by this inventory.
    """
    produced = set()
    for log in sorted(FIX.parent.rglob("*.log")):
        produced |= set(servediff.parse(log.read_text(encoding="utf-8"))["facts"])
    expected = {
        line.strip()
        for line in (FIX / "expected_fact_keys.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    missing = sorted(expected - produced)
    added = sorted(produced - expected)
    assert missing == [], f"fact keys no longer produced: {missing}"
    assert added == [], (
        f"new fact keys {added} — add a test that pins them, then run "
        "`python tools/fact_keys.py --write`"
    )


def test_the_sglang_reports_audit_to_zero_and_yield_the_mamba_facts(capsys):
    # Six public SGLang reports, reassembled verbatim (PROVENANCE.md says which code blocks).
    # They are the SGLang half of the "audit is honest" claim: before this pass the mamba
    # cache, the concurrency cap it forces and the per-rank effective size were all just
    # lines this tool handed back as "worth a look".
    d = FIX / "sglang-2026-10"
    assert servediff.main(["audit", str(d), "--max-shapes", "0"]) == 0
    out = capsys.readouterr().out
    assert "6 log(s): 0 with shapes worth a look" in out

    f = facts("sglang-2026-10/sglang-41654.log")
    assert f["mamba.cache_size"].value == 63
    assert f["mamba.state_gib"].value == "0.18/9.00"
    assert f["sched.max_running_requests"].value == (
        "12 (capped by mamba cache: 63 slots, 5 per request)")

    f = facts("sglang-2026-10/sglang-46756.log")
    # Two ranks in one launch, two different answers — a single fact would have to pick one.
    sizes = f["mamba.effective_size"].value
    assert isinstance(sizes, list) and len(sizes) == 2
    assert set(sizes) == {"72 (raised from 24, needed 72 on DP3/TP3/EP3)",
                          "18 (raised from 6, needed 18 on DP7/TP7/EP7)"}
    assert f["scheduler.exit"].value == "code -3 (reason 11)"

    f = facts("sglang-2026-10/sglang-43741.log")
    assert f["kv_cache.fp8_scaling"].value == "defaulted to 1.0. (accuracy risk)"


def test_a_prompt_prefixed_line_is_a_command_not_a_setting():
    # `$ PYTHONPATH=<main @ …> python3 -m sglang.launch_server --mem-fraction-static 0.85`
    # (sglang#52318) carries six flag-looking tokens that never reached the server.
    from servediff import audit
    assert audit.is_noise("$ PYTHONPATH=<main @ 448764da63>/python python3 -m sglang.launch_server --port 30000 --mem-fraction-static 0.85")
    assert audit.is_noise("$ vllm serve Qwen/Qwen3-8B --max-model-len 32768")
    assert audit.is_noise("vllm serve Qwen/Qwen3-8B --max-model-len 32768")
    # A log line that merely mentions a path is not a prompt.
    assert not audit.is_noise("INFO 11-22 [engine.py:42] loaded $PROFILE from /root/model")


def test_a_384_thread_cpu_forum_report_is_read_without_a_ready_line():
    # discuss.vllm.ai thread 2708: 384 threads, 1.5 TB RAM, and the poster sees two threads
    # busy. The paste is a fragment (no ready line), which is the normal shape of a forum
    # post; the facts still have to come out, and `audit` still has to say zero.
    f = facts("vllm-forum-2026-06/vllm-cpu-2x-epyc-9655.log")
    assert f["cpu.core_ids"].value.startswith("[192, 193")
    assert f["cpu.core_ids"].value.count(",") == 94
    assert f["cpu.reserved_cpus"].value == "[287]"
    assert f["cpu.threads_bind_mode"].value == "auto (auto_setup=True, skip_setup=False)"
    assert f["kv_cache.cpu_node"].value == "675.87/755.19 GiB (node 0, auto)"
    assert f["kv_cache.requested_gib"].value == 694.77
    assert f["kv_cache.non_kv_gib"].value == 18.9
    assert f["kv_cache.tokens"].value == 4921472
    assert f["kv_cache.max_concurrency"].value == 150.19
    # Triton finding no driver is why this run is on the v1 model runner.
    assert f["triton.drivers"].value == "0 active (Triton disabled)"
    assert f["model_runner"].value == "v1 (v2 requires Triton)"
    # A fragment: the absence is honest, not inferred.
    assert "startup" not in f


def test_a_pd_disaggregation_crash_still_names_the_config_it_died_with():
    # discuss.vllm.ai thread 2637. A prefill/decode pair over P2P NCCL hangs in the 60 s
    # broadcast wait and then dumps the whole VllmConfig. That dump is one long sentence, so
    # before this rule the single most useful line in the paste was classified as prose.
    f = facts("vllm-forum-2026-05/pd-nccl-sample-tokens-timeout.log")
    assert f["engine.version"].value == "0.20.1"
    assert f["weights.model"].value == "/data-new/models/Qwen3-30B-A3B"
    assert f["weights.dtype"].value == "bfloat16"
    assert f["kv_cache.max_seq_len"].value == 8192
    assert f["parallel.tp"].value == 2
    assert f["parallel.pp"].value == 1
    assert f["parallel.dp"].value == 1
    assert f["cache.prefix"].value == "enabled"
    assert f["cache.chunked_prefill"].value == "enabled"
    assert f["compile.backend"].value == "inductor"
    # Only the non-None NCCL variables were overridden; the rest are the ones it may set.
    assert f["nccl.env_overrides"].value == "{NCCL_BUFFSIZE=4194304}"
    assert f["startup.cause"].value == "TimeoutError: RPC call to sample_tokens timed out."
    # It failed, and it never reached ready; "failed" twice is the output-handler path plus
    # the engine itself, and neither value claims a working server.
    assert "ready" not in f["startup"].values


def test_a_request_field_the_server_ignored_is_a_fact():
    # thread 2426: a tool parser was imported and `enable_thinking` was accepted then dropped
    # from the request. Neither appears in any flag, and both change what the model saw.
    f = facts("vllm-forum-2026-05/broadcast-60s-tool-parser.log")
    assert f["api.tool_parser"].value == "Qwen3CoderToolParser"
    assert f["api.ignored_request_fields"].value == "{'enable_thinking'}"
    assert f["engine.version"].value == "0.14.0"


def test_speculative_decoding_settings_that_look_active_are_facts():
    # thread 2946: vLLM 0.28, Qwen3.5-27B-NVFP4, run once plain and once with MTP spec decode.
    # Three settings a user believes are in force are overridden and only mentioned in warnings:
    # the token budget is reset, min_p/logit_bias stop working, and the MTP layer is re-forwarded
    # per draft token. The kernels are the other half: a flag names the format, never the kernel.
    f = facts("vllm-forum-2026-09/qwen35-nvfp4-mtp.log")
    assert f["sched.max_num_scheduled_tokens"].value == 2048
    assert f["sampling.ignored_with_spec"].value == "min_p, logit_bias"
    assert f["spec.mtp_forward_repeat"].value == "num_spec_tokens > 1"
    assert f["spec.mtp_weights_shared"].values == ["embedding", "lm_head"]
    assert f["quant.kernel.nvfp4"].value == "FlashInferCutlassNvFp4LinearKernel"
    assert f["quant.kernel.selected"].value == (
        "CompressedTensorsW8A8Fp8: CutlassFP8ScaledMMLinearKernel")
    assert f["compile.deep_gemm"].value == "vendored (vllm.third_party)"
    assert f["compile.deep_gemm_feature"].values == ["PDL enabled", "E8M0 enabled"]
    assert f["gdn.decode_kernel"].value == "cuda"
    assert "Triton/FLA" in f["gdn.prefill_kernel"].value
    assert f["mm.prefix_attention"].value == "disabled (multimodal config off)"


def test_a_log_pasted_through_a_forum_still_parses():
    # The forum returns the post with typographic quotes: `is set to ‘align’` and `won’t`.
    # The server's log is ASCII, so every rule is written in ASCII; without curling the input
    # back, a pasted copy loses the mamba cache mode and the spec warning silently.
    text = (
        "(EngineCore pid=1) INFO 09-08 08:23:42 [config.py:605] Mamba cache mode is set to "
        "\u2018align\u2019 for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled\n"
        "(EngineCore pid=1502) WARNING 09-08 09:41:56 [init.py:205] min_p and logit_bias "
        "parameters won\u2019t work with speculative decoding.\n"
    )
    f = servediff.parse(text)["facts"]
    assert f["kv_cache.mamba_mode"].value == "align (default for Qwen3_5ForConditionalGeneration)"
    assert f["sampling.ignored_with_spec"].value == "min_p, logit_bias"


def test_a_pure_traceback_says_what_it_is_instead_of_exiting_silently():
    # sglang#52318 is a stack with no startup lines at all. "not recognised" alone reads as
    # a bug in the reader; the hint names the difference and says there is nothing to compare.
    log = Path(__file__).parent / "fixtures" / "sglang-2026-11" / "sglang-52318-assertion.log"
    out = subprocess.run([sys.executable, "-m", "servediff", "show", str(log)],
                         capture_output=True, text=True)
    assert out.returncode == 2
    assert "no vLLM or SGLang configuration lines recognised" in out.stderr
    assert "this looks like a traceback, not a startup log" in out.stderr
    assert "31 line(s) of stack" in out.stderr


def test_a_hybrid_model_rewrites_its_cache_dtype_and_pays_for_a_padding_layer():
    # thread 2254: vLLM 0.14 on Nemotron-3-Nano-30B-A3B, TP4 over four V100s. Two things the
    # command line never says: the engine moves the mamba ssm dtype itself, and it inserts a
    # padding layer whose cost is printed as a percentage of the whole KV cache.
    f = facts("vllm-forum-2026-01/nemotron3-nano-v100-tp4.log")
    assert f["engine.version"].value == "0.14.0"
    assert f["model.architecture"].value == "NemotronHForCausalLM"
    assert f["kv_cache.mamba_ssm_dtype"].value == "float16 (set for NemotronH)"
    assert f["kv_cache.padding_layers"].value == 1
    assert f["kv_cache.padding_waste"].value == "4.35%"
    assert f["moe.config"].value == "default (config file not found)"
    assert f["parallel.dp_sync_nccl"].value == "disabled (async scheduling)"
    assert f["scheduler.async"].value == "enabled"


def test_a_pre_ampere_card_says_what_it_cast_and_which_kernel_it_refused():
    # Same log: bfloat16 is unavailable on V100, so the weights are cast down and FA2 refuses
    # to run; custom allreduce also switches off, and it names the reason (PCIe-only box).
    f = facts("vllm-forum-2026-01/nemotron3-nano-v100-tp4.log")
    assert f["gpu.bf16_support"].value == "no (Tesla V100-SXM2-32GB)"
    assert f["model.cast"].value == "torch.bfloat16 -> torch.float16"
    assert f["backend.flash_attn_blocked"].value == "FA2 needs sm80+"
    assert f["backend.allreduce_reason"].value == (
        "it's not supported on more than two PCIe-only GPUs")


def test_the_local_rank_printed_after_api_server_is_not_a_port():
    # `Starting vLLM API server 0 on http://0.0.0.0:8000`: the 0 is the local rank inside a
    # data-parallel group, not a port. Reading it as a port produced `server.port: 0` on a TP4
    # box, which then differed from every other log for no reason. The URL carries the port.
    f = facts("vllm-forum-2026-01/nemotron3-nano-v100-tp4.log")
    assert f["server.url"].value == "http://0.0.0.0:8000"
    assert "server.port" not in f


def test_a_log_pasted_through_a_forum_keeps_the_two_hyphens_of_a_cache_path():
    # The forum's typography turns Hugging Face's `models--Qwen--Qwen3.6` into an en dash, so a
    # pasted path reads `models\u2013Qwen\u2013Qwen3.6` and can never equal the server's path.
    # Folding en/em dashes back to `--`/`---` is the same class of fix as the curly quotes.
    text = (
        "(APIServer pid=1) INFO 04-20 06:08:21 [utils.py:233] non-default args: "
        "{'model': '/root/.cache/huggingface/hub/models\u2013Qwen\u2013Qwen3.6-35B-A3B-FP8'}\n"
    )
    f = servediff.parse(text)["facts"]
    assert f["arg.model"].value == (
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8")
    assert "\u2013" not in f["arg.model"].value


def test_an_eager_run_and_a_kernel_config_fallback_are_facts():
    # thread 2570: vLLM 0.19, Qwen3.6-35B-A3B-FP8 on one GPU with enforce_eager. The GDN
    # prefill line has no `(requested=…, head_k_dim=…)` suffix in this build, and the FP8
    # kernel config falls back to its default because no config file exists.
    f = facts("vllm-forum-2026-04/qwen36-35b-fp8-eager.log")
    assert f["cuda_graph"].value == "disabled (enforce_eager)"
    assert f["compile"].value == "disabled (enforce_eager)"
    assert f["gdn.prefill_kernel"].value == "Triton/FLA"
    assert f["quant.kernel.config"].value == "default W8A8 (no config file)"
    assert f["arg.model"].value.startswith(
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/")


def test_a_ray_executor_turns_async_scheduling_off_and_keeps_the_reason():
    # thread 2526: four nodes under Ray. `--async-scheduling` is accepted and then turned off,
    # the TP size is larger than the GPUs it can place, and the run dies in a worker KeyError.
    # The reason is its own key so a diff can tell "flag off" from "backend refused".
    f = facts("vllm-forum-2026-04/ray-4-node-random-fail.log")
    assert f["scheduler.async"].value == "disabled"
    assert f["scheduler.async_reason"].value == "ray executor"
    assert f["parallel.tp_exceeds_gpus"].value == "tp 16 > gpus 4"
    assert f["parallel.world_size"].value == 16
    assert "'NCCL_'" in f["ray.env_prefixes"].value
    assert "ray.exceptions.RayTaskError(KeyError)" in f["startup.error"].values


def test_a_model_with_two_head_widths_forces_one_attention_backend():
    # thread 2674: Gemma-4-31B-IT-NVFP4 on 2x16GB. Local heads are 256 wide, global heads 512,
    # so the engine overrides the backend to keep both on the same kernel set. The checkpoint
    # is ModelOpt's layout, and the model runs text-only because every modality limit is 0.
    f = facts("vllm-forum-2026-05/gemma4-nvfp4-hetero-heads.log")
    assert f["engine.version"].value == "0.21.0"
    assert f["backend.attention_forced"].value == "TRITON_ATTN (head_dim 256 vs 512)"
    assert f["quant.checkpoint_format"].value == "ModelOpt NVFP4 (NVFP4), experimental"
    assert f["mm.text_only"].value == "all modality limits 0"
    assert f["mm.chunked_input"].value == "disabled (multimodal-bidirectional attention)"


def test_the_oldest_log_in_the_corpus_still_yields_its_rank_line():
    # thread 1602: vLLM 0.10.1.dev1, before PCP rank existed. The rule takes whatever rank
    # list the build prints instead of pinning the set, so this line is read rather than lost.
    f = facts("vllm-forum-2025-09/mistral-7b-huge-batched-tokens.log")
    assert f["parallel.roles"].value == "DP rank 0, PP rank 0, TP rank 0, EP rank 0"
    assert f["engine.platform"].value == "cuda"
    assert f["backend.attention"].value == "Flash Attention (V1 engine)"
    assert f["weights.format"].value == "'*.safetensors'"
    assert f["weights.download_s"].value == 17.724022


def test_a_vram_is_enough_thread_names_the_process_holding_the_card():
    # thread 1234: vLLM 0.9.0, TP4, OOM on GPU 3 while the card looks free. The older wording
    # prints the holder: 94.41 of 95.00 GiB is another process, and vLLM asked for 768 MiB of
    # the remaining 600. The pid is deliberately not a fact — it changes every run.
    f = facts("vllm-forum-2025-07/qwen-vram-enough-but-oom.log")
    assert f["engine.version"].value == "0.9.0"
    assert f["torch.oom_device"].value == 3
    assert f["torch.oom_requested_mib"].value == 768.0
    assert f["torch.oom_free_mib"].value == 600.06
    assert f["torch.oom_total_gib"].value == 95.0
    assert f["torch.oom_holder_gib"].value == 94.41
    assert "oom_holder_pid" not in f


def test_a_plugin_group_and_a_multitask_default_are_facts():
    # Same log: this is the build that still loads a plugin group, and the only record of what
    # runs before the model is the list it prints. A five-task checkpoint served with no task
    # flag also says which one it defaulted to.
    f = facts("vllm-forum-2025-07/qwen-vram-enough-but-oom.log")
    assert f["plugins.loaded"].value == (
        "lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver")
    assert f["model.tasks"].value.startswith("generate (default of 'generate', 'classify'")
    assert f["sampling.topk_topp_fallback"].value == "torch-native (FlashInfer unavailable)"


def test_the_sixty_second_wait_is_one_wait_relogged_on_two_a2_cards():
    # thread 2828: TP2 on A2, six wait lines exactly 60.000 s apart after graph capture had
    # already finished. The numbers that rule out "still compiling" are the capture time and
    # the KV budget; `repeats` turns the cadence into one line.
    f = facts("vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log")
    assert f["kv_cache.memory_gib"].value == 4.91
    assert f["kv_cache.tokens"].value == 407080
    assert f["cuda_graph.capture_s"].value == 4
    assert f["model_loading.gib"].value == 8.29
    out = subprocess.run([sys.executable, "-m", "servediff", "repeats",
                          str(FIX / "vllm-forum-2026-07" / "a2-stuck-on-shm-broadcast.log"),
                          "--top", "3"], capture_output=True, text=True)
    assert "6x regular" in out.stdout
    assert "60.000 s median" in out.stdout
    # The timestamp on a wait line is the end of the wait, so the first one began at 09:02:37 —
    # 3 s before a capture that finished at 09:02:40 started waiting on nothing.
    assert "began at 09:02:37" in out.stdout
    assert "0 other line(s) inside that span" in out.stdout


def test_audit_of_a_corpus_directory_reaches_the_dated_subdirectories(tmp_path):
    # The corpus keeps every batch in its own `vllm-forum-2026-07/` style folder. A one-level
    # glob audited 21 of 86 logs and still printed a count, which is the silent-scope failure
    # this tool is supposed to catch; the count now equals what an rglob finds.
    nested = tmp_path / "batch-2026-07"
    nested.mkdir()
    (nested / "one.log").write_text("INFO 07-01 10:00:00 [core.py:1] hello\n")
    (tmp_path / "flat.log").write_text("INFO 07-01 10:00:00 [core.py:1] hello\n")
    expected = len([p for p in sorted(tmp_path.rglob("*.log"))])
    assert expected == 2
    out = subprocess.run([sys.executable, "-m", "servediff", "audit", "--max-shapes", "0",
                          str(tmp_path)], capture_output=True, text=True)
    assert f"{expected} log(s):" in out.stdout


def test_the_whole_fixture_corpus_is_audited_not_a_slice_of_it():
    logs = sorted(FIX.rglob("*.log"))
    assert len(logs) > 50, "the corpus should be the many-directory kind this guards"
    out = subprocess.run([sys.executable, "-m", "servediff", "audit", "--max-shapes", "0",
                          str(FIX)], capture_output=True, text=True)
    assert f"{len(logs)} log(s):" in out.stdout
    assert "0 with shapes worth a look" in out.stdout



def test_a_wait_that_began_when_a_kernel_began_compiling_is_said_in_one_line():
    # vllm#45198: the JIT warning is at 22:01:40, the first wait line at 22:02:40. A wait line
    # is the end of a 60 s wait, so the wait began at the same second the kernel started
    # compiling — that is the whole diagnosis, and it is arithmetic on two timestamps.
    f = facts("vllm-issue-2026-06/qwen3.5-tp2-jit-then-wait.log")
    assert f["kernel.jit_during_inference"].value == "_zero_kv_blocks_kernel"
    out = subprocess.run([sys.executable, "-m", "servediff", "repeats",
                          str(FIX / "vllm-issue-2026-06" / "qwen3.5-tp2-jit-then-wait.log")],
                         capture_output=True, text=True)
    assert "3x regular  22:02:40 -> 22:04:40" in out.stdout
    assert "began at 22:01:40" in out.stdout
    assert "0 other line(s) inside that span" in out.stdout


def test_show_points_at_repeats_when_the_cadence_is_the_problem():
    # A hang log answers `show` with a dozen normal startup facts; without a pointer the
    # reader never runs the command that reads the cadence. The pointer goes to stderr so
    # `servediff show log | grep` stays pipeable.
    out = subprocess.run([sys.executable, "-m", "servediff", "show",
                          str(FIX / "vllm-forum-2026-07" / "a2-stuck-on-shm-broadcast.log")],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "servediff repeats <log>" in out.stderr
    assert "6 times at a 60.000 s cadence" in out.stderr
    assert "servediff repeats <log>" not in out.stdout
    # A log that repeats nothing gets no pointer.
    quiet = subprocess.run([sys.executable, "-m", "servediff", "show",
                            str(FIX / "vllm-47047-v0.23-marlin.log")],
                           capture_output=True, text=True)
    assert "servediff repeats <log>" not in quiet.stderr


def test_sglang_staging_buffers_and_its_library_block_are_facts():
    # SGLang's PD-disaggregation path allocates KV staging buffers line by line; four 896 MiB
    # allocations are 3.5 GiB of the card that no argument mentions. The same block says it
    # fell back to plain cudaMalloc, and its library list marks torchao as missing — a code
    # path downgrade that is not an error and leaves no other trace.
    f = facts("sglang-2026-10/sglang-42684.log")
    assert f["transfer.staging_buffer_mib"].value == 896.0
    assert f["transfer.staging_buffers"].value == 4
    assert f["transfer.staging_total_mib"].value == 3584.0
    assert f["transfer.staging_method"].value == "cudaMalloc"
    assert f["transfer.custom_mem_pool"].value == "disabled (cudaMalloc)"
    assert f["env.lib.torchao"].value == "Module Not Found"
    assert f["env.lib.flashinfer_jit_cache"].value == "0.6.18+cu130"
    assert f["gpu.compute_capability"].value == "9.0"


def test_a_utf16_log_is_read_instead_of_mojibake(tmp_path):
    # PowerShell's `>` and `Out-File` write UTF-16LE with a BOM. Decoded as UTF-8 that is
    # NUL-separated mojibake: "no configuration lines recognised" on a perfectly good log,
    # which is the first thing a Windows user would hit. The bytes are sniffed instead.
    source = (FIX / "vllm-forum-2026-07" / "a2-stuck-on-shm-broadcast.log").read_text(
        encoding="utf-8")
    utf16 = tmp_path / "redirect.log"
    utf16.write_bytes(source.encode("utf-16"))
    out = subprocess.run([sys.executable, "-m", "servediff", "show", str(utf16)],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "read as utf-16" in out.stderr
    assert "kv_cache.memory_gib" in out.stdout
    assert "4.91" in out.stdout


def test_utf16_on_stdin_is_decoded_too():
    source = (FIX / "vllm-issue-2026-06" / "qwen3.5-tp2-jit-then-wait.log").read_bytes();
    # Bytes, so the test's own encoding cannot influence what arrives on stdin.
    out2 = subprocess.run([sys.executable, "-m", "servediff", "show", "-"],
                          input=source.decode("utf-8").encode("utf-16"),
                          capture_output=True)
    assert out2.returncode == 0
    # Compare bytes: on a Windows runner the child writes stderr in the ANSI code page, so
    # decoding it as UTF-8 in the test was what failed, not the note itself.
    assert b"read as utf-16" in out2.stderr
    assert b"kernel.jit_during_inference" in out2.stdout


def test_the_short_log_hint_counts_its_lines():
    # The hint used to be a plain string with a `{}` placeholder in it, so a 2-line paste
    # was told "only {} line(s) here".
    out = subprocess.run([sys.executable, "-m", "servediff", "show", "-"],
                         input="hello\nworld\n", capture_output=True, text=True)
    assert out.returncode == 2
    assert "only 2 line(s) here" in out.stderr
    assert "{}" not in out.stderr


def test_a_gzipped_log_is_read_without_zcat(tmp_path):
    # Rotated logs stay gzipped on disk. Answering "no configuration lines recognised" about
    # a .gz of a perfectly good log sends people to `zcat` and then back again.
    import gzip as gz
    source = (FIX / "vllm-forum-2026-07" / "a2-stuck-on-shm-broadcast.log").read_bytes()
    packed = tmp_path / "serve.log.gz"
    packed.write_bytes(gz.compress(source))
    out = subprocess.run([sys.executable, "-m", "servediff", "show", str(packed)],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "gzip-compressed; decompressed" in out.stderr
    assert "kv_cache.memory_gib" in out.stdout
    assert "4.91" in out.stdout


def test_show_on_a_directory_points_at_audit():
    # `Is a directory` is a strerror, not an answer: reading a directory as one corpus is
    # what `audit` does, and the message should say so.
    out = subprocess.run([sys.executable, "-m", "servediff", "show", str(FIX)],
                         capture_output=True, text=True)
    assert out.returncode == 2
    assert "is a directory" in out.stderr
    assert "servediff audit" in out.stderr


def test_audit_of_a_directory_includes_gzipped_logs(tmp_path):
    import gzip as gz
    source = (FIX / "vllm-forum-2026-07" / "a2-stuck-on-shm-broadcast.log").read_bytes()
    (tmp_path / "plain.log").write_bytes(source)
    (tmp_path / "rotated.log.gz").write_bytes(gz.compress(source))
    out = subprocess.run([sys.executable, "-m", "servediff", "audit", str(tmp_path)],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "rotated.log.gz" in out.stdout
    assert "2 log(s)" in out.stdout


def test_a_bzip2_log_says_what_it_is_instead_of_that_it_has_no_config(tmp_path):
    import bz2
    source = (FIX / "vllm-forum-2026-07" / "a2-stuck-on-shm-broadcast.log").read_bytes()
    packed = tmp_path / "serve.log.bz2"
    packed.write_bytes(bz2.compress(source))
    out = subprocess.run([sys.executable, "-m", "servediff", "show", str(packed)],
                         capture_output=True, text=True)
    assert out.returncode == 2
    assert "bzip2-compressed" in out.stderr
    assert "bunzip2" in out.stderr
    assert "no vLLM or SGLang configuration lines" not in out.stderr
