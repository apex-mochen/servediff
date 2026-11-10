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
    # Silent gates inside startup: JIT compile time and whether late compiles are loud.
    assert f["jit.warmup_s"].value == 0.0
    assert f["jit.monitor_mode"].value == "warn"
    assert f["compile.autotune_gemm"].value == "disabled (not enough SMs)"
    # torch.compile is one number in the summary line (9.82 s) and three parts here; an
    # upgrade that doubles startup is usually one of these three moving, not all of them.
    assert f["compile.total_s"].value == 9.82
    assert f["compile.dynamo_s"].value == 3.61
    assert f["compile.range"].value == "1,2048"
    assert f["compile.graph_s"].value == 4.17
    assert f["compile.warmup_s"].value == 0.13
    # Serving torch threads are cut, and the pair is the setting: "1" alone would not say
    # whether the cut came from 16 or from 4.
    assert f["cpu.torch_threads"].value == "16 -> 1"
    # Sampling defaults came from the model, not from the flags.
    assert "repetition_penalty" in f["sampling.overridden"].value
    # One fact per op: the whole IrOpPriorityConfig line is too wide for the table and
    # two different configs would truncate to the same visible string.
    assert f["kernel.ir_op.rms_norm"].value == "native"
    assert f["kernel.ir_op.gelu_and_mul_sparse"].value == "triton|native"


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
    # The claim "twenty logs from public reports, nothing worth a look" should be
    # checkable in one command rather than twenty.
    logs = sorted(FIX.glob("*.log"))
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


def test_the_tp8_report_log_leaves_nothing_unread_because_these_facts_are_read():
    # Every line below is a shape the audit named as "no rule reads them" on a real
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
    assert f["compile.cached_range"].value == "1,102"
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
    assert len(notes) == 1 and "world_size=1" in notes[0] and "TP>1" in notes[0]


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
        assert any("DEBUG" in n for n in notes) is expected


def test_markdown_reports_absence_notes(tmp_path, capsys):
    body = ("INFO [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='m'\n"
            "INFO [parallel_state.py:1827] world_size=1 rank=0 local_rank=0\n")
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(body, encoding="utf-8")
    b.write_text(body, encoding="utf-8")
    assert servediff.main(["diff", str(a), str(b), "--markdown"]) == 0
    out = capsys.readouterr().out
    assert "ℹ️" in out and "world_size=1" in out
