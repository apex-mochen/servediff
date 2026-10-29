# Fixtures

Log lines copied from public issue threads, one file per side of each comparison.
They are excerpts the reporters chose to post, not full logs: the tests prove the
parser recognises these formats, not that it finds them inside a full noisy log.

| File | Source (checked 2026-10-28) |
|---|---|
| `vllm-47047-v0.23-default.log` | vllm-project/vllm#47047 body, "Notes from the logs": v0.23.0 default selection |
| `vllm-47047-v0.23-marlin.log` | same, v0.23.0 with `--moe-backend marlin` |
| `vllm-53472-v0.23.log` | vllm-project/vllm#53472 body, "Runtime evidence", v0.23.0 (line wrap kept as posted) |
| `vllm-53472-v0.26.log` | same, v0.26.0 |
| `vllm-53472-v0.26-flag.log` | same issue, "Ablation": v0.26.0 + `--moe-backend flashinfer_cutlass` (lines quoted inline in the issue) |
