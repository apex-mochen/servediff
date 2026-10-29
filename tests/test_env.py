import json
import sys

from servediff import cli, env


def snap(packages, env_vars=None, gpus=None, python="3.12.3"):
    return {"schema": env.SCHEMA, "python": python, "packages": packages,
            "env": env_vars or {}, "gpus": gpus or []}


def test_snapshot_of_this_interpreter(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_API_KEY", "should-not-leak")
    monkeypatch.setenv("UNRELATED_VAR", "x")
    data = env.snapshot(sys.executable)
    assert data["schema"] == env.SCHEMA
    assert "pytest" in data["packages"]
    assert data["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert data["env"]["VLLM_API_KEY"] == "<redacted>"
    assert "UNRELATED_VAR" not in data["env"]
    assert "should-not-leak" not in json.dumps(data)


def test_diff_reports_version_changes_and_ranks_key_packages_first():
    a = snap({"torch": "2.11.0", "triton": "3.6.0", "aiohttp": "3.9", "vllm": "0.28.0"})
    b = snap({"torch": "2.11.0", "triton": "3.7.1", "aiohttp": "3.10", "vllm": "0.30.0", "new-pkg": "1"})
    rows = env.ranked(env.diff(a, b))
    keys = [r["key"] for r in rows]
    assert "pkg.torch" not in keys  # unchanged
    assert keys.index("pkg.triton") < keys.index("pkg.aiohttp")
    assert {r["key"]: r["status"] for r in rows}["pkg.new-pkg"] == "only_candidate"


def test_cli_diffs_two_snapshots_and_rejects_mixed_inputs(tmp_path, capsys):
    a, b, log = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "s.log"
    a.write_text(json.dumps(snap({"flashinfer-python": "0.6.16.post3"})))
    b.write_text(json.dumps(snap({"flashinfer-python": "0.6.18.post1"})))
    log.write_text("Using FLASH_ATTN attention backend\n")
    assert cli.main(["diff", str(a), str(b)]) == 1
    assert "0.6.18.post1" in capsys.readouterr().out
    c = tmp_path / "c.json"
    c.write_text(json.dumps(snap({"flashinfer-python": "0.6.16.post3", "mypy": "1.0"})))
    assert cli.main(["diff", str(a), str(c)]) == 1
    out = capsys.readouterr().out
    assert "pkg.mypy" not in out and "1 in other packages (--all to list them)" in out
    assert cli.main(["diff", str(a), str(c), "--all"]) == 1
    assert "pkg.mypy" in capsys.readouterr().out
    assert cli.main(["diff", str(a), str(a)]) == 0
    assert cli.main(["diff", str(a), str(log)]) == 2


def test_env_command_writes_file(tmp_path):
    out = tmp_path / "env.json"
    assert cli.main(["env", "--python", sys.executable, "-o", str(out)]) == 0
    assert env.is_snapshot(out.read_text())


def test_env_command_reports_bad_interpreter(tmp_path):
    assert cli.main(["env", "--python", str(tmp_path / "no-such-python")]) == 2
