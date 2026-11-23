# “min_p and logit_bias parameters won’t work with speculative decoding” — three settings spec decode changes without saying so in the flags

vLLM prints this warning once, at startup, per process:

```
WARNING [init.py:205] min_p and logit_bias parameters won't work with speculative decoding.
```

Two more warnings come with it, and all three describe the same thing: flags you passed are
still in the command line, still in `non-default args`, and no longer describe the run.

| what you set | what the run does | the line that says so |
|---|---|---|
| `min_p`, `logit_bias` | ignored for every request | `[init.py:205] min_p and logit_bias parameters won't work with speculative decoding` |
| `--max-num-batched-tokens 4096` | the scheduled-token budget is reset to 2048 | `[vllm.py:1862] max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings` |
| `--num-speculative-tokens 3` | the same MTP layer is forwarded three times per step | `[speculative.py:971] Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer` |

The middle one is the expensive one: the reset budget explains a drop in throughput after
turning speculation on, and the flag that looks responsible (`--max-num-batched-tokens`) is
exactly the one that lost. The third is a draft-acceptance/throughput trade nobody measures
because nothing in the serving log after startup mentions it again.

All of the above was read from one forum post, not from the source: `discuss.vllm.ai/t/2946`,
vLLM 0.28.0, Qwen3.5-27B-NVFP4 served twice — once plain, once with
`--speculative-config method=mtp num_speculative_tokens=2`. The paste is a 86-line startup
log, and `servediff 0.4.65` reads it to:

```
$ servediff show qwen35-nvfp4-mtp.log
compile.deep_gemm                vendored (vllm.third_party)                                  line 20
compile.deep_gemm_feature        ["PDL enabled", "E8M0 enabled"]                              line 21
gdn.decode_kernel                cuda                                                         line 24
gdn.prefill_kernel               Triton/FLA (requested auto, head_k_dim 128)                  line 23
kv_cache.mamba_mode              align (default for Qwen3_5ForConditionalGeneration)          line 10
mm.prefix_attention              disabled (multimodal config off)                             line 45
model.architecture               ["Qwen3_5ForConditionalGeneration", "Qwen3_5MTP"]            line 8
quant.kernel.nvfp4               FlashInferCutlassNvFp4LinearKernel                           line 25
quant.kernel.selected            CompressedTensorsW8A8Fp8: CutlassFP8ScaledMMLinearKernel     line 19
sampling.ignored_with_spec       min_p, logit_bias                                            line 54
sched.max_num_scheduled_tokens   2048                                                         line 49
spec.mtp_forward_repeat          num_spec_tokens > 1                                          line 46
spec.mtp_weights_shared          ["embedding", "lm_head"]                                     line 74
```

Diff that against the same server without speculation and the three rows above are the
difference between "the flags are the same" and "the run is the same".

Two other things in the same paste, for the record:

- The quantised GEMM kernels are named per format and per model: `FlashInferCutlassNvFp4LinearKernel`
  for the NVFP4 layers, `CutlassFP8ScaledMMLinearKernel` for the FP8 ones
  (`CompressedTensorsW8A8Fp8`). `--quantization compressed-tensors` names the format, never the
  kernel, and the kernel is where a throughput delta between two runs on one checkpoint comes
  from.
- DeepGEMM fell back to the vendored copy (`deep_gemm not found in site-packages`) with PDL and
  E8M0 enablement, none of which is a flag either.

The fixture is `tests/fixtures/vllm-forum-2026-09/qwen35-nvfp4-mtp.log` with a `PROVENANCE.md`
that says how the forum's paragraph-per-line rendering was turned into a file, including the
typographic quotes the page serves (`‘align’`, `won’t`) — `servediff` curls those back to ASCII
before matching, because a copy from a browser or a chat client is what people actually paste.
That file audits to zero shapes worth a look.

One limit worth stating: this says the budget was reset, not why 2048 was chosen. The value
comes from the spec config and the model's max context; the log reports the result, and only a
second run with a different `--max-num-batched-tokens` tells you whether the reset is binding.
