import io
import json
import re
import subprocess
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
