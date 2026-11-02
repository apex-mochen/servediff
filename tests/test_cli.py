import json
import sys
from pathlib import Path

from servediff import cli as servediff

FIX = Path(__file__).with_name("fixtures")


def facts(name: str) -> dict:
    return servediff.parse((FIX / name).read_text(encoding="utf-8"))["facts"]


def changed(a: str, b: str) -> dict:
    rows = servediff.diff(servediff.parse((FIX / a).read_text()), servediff.parse((FIX / b).read_text()))
    return {r["key"]: (r["baseline"], r["candidate"]) for r in rows if r["status"] != "same"}


# --- Public issue excerpts -------------------------------------------------------

def test_47047_moe_backend_difference_is_found():
    # The difference the reporter found by hand in vllm#47047.
    assert changed("vllm-47047-v0.23-default.log", "vllm-47047-v0.23-marlin.log") == {
        "backend.moe": ("VLLM_CUTLASS (NvFp4)", "MARLIN (NvFp4)"),
    }


def test_53472_version_swap_is_found():
    diff = changed("vllm-53472-v0.23.log", "vllm-53472-v0.26.log")
    assert diff["backend.moe"] == ("FlashInfer CUTLASS (Unquantized)", "TrtLlmBf16ExpertsMonolithic")
    # v0.23 excerpt has no prepare/finalize line: absent, not assumed.
    assert diff["backend.moe_prepare_finalize"] == ("<absent>", "MoEPrepareAndFinalizeNoDPEPMonolithic")


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
