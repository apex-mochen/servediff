# servediff（中文说明）

**参数一样，server 却不一样。** `servediff` 告诉你 vLLM / SGLang server **实际**用了什么配置，以及两次运行之间哪里变了。

benchmark 结果文件只记录客户端。升级之后，server 可能会悄悄换掉 MoE 或 attention backend、改变 CUDA graph 的捕获方式、走另一条回退路径，或者底下的 torch / triton / flashinfer 版本已经不同。现在要发现这些，只能人工读启动日志、逐个 bisect。servediff 替你把日志读一遍。

- **适合谁**：比较两次 vLLM / SGLang 运行的人，比如升级前后、改了某个参数、换了机器。
- **读什么**：server 的启动日志；也可以读一份 server Python 环境的快照。离线运行，不需要 GPU，没有任何依赖。
- **不做什么**：不判断性能变化是哪一项造成的，那仍然需要 A/B 实验。它告诉你该 A/B 哪几项。

## 一分钟试用（不需要 GPU）

```bash
git clone https://github.com/apex-mochen/servediff && cd servediff
pip install .
servediff diff examples/issue-excerpts/vllm-47047-v0.23-default.log examples/issue-excerpts/vllm-47047-v0.23-marlin.log
```

输出只有一行差异：MoE backend 从 `VLLM_CUTLASS (NvFp4)` 变为 `MARLIN (NvFp4)`。这正是 [vllm#47047](https://github.com/vllm-project/vllm/issues/47047) 的报告者在 v0.21 → v0.23 吞吐下降后，靠人工找出来的差异。

## 真实案例：参数相同，vLLM 0.28.0 对 0.30.0

在一台 RTX 4060 Laptop（WSL2）上，两个版本用完全相同的参数启动：

- 0.28.0 默认配置**起不来**，报 `RuntimeError: UVA is not available`。
- 0.30.0 能起来，因为 pinned memory 不可用时会回退到 device memory。上游修复是 [vllm#56908](https://github.com/vllm-project/vllm/pull/56908)，v0.30.0 是第一个包含它的正式版。

servediff 会先给出"baseline 启动失败"的 warning 和根因，再列出两边的配置差异。完整输出见英文 README。

## 依赖版本：`servediff env`

```bash
servediff env --python /旧环境/bin/python -o old.json
servediff env --python /新环境/bin/python -o new.json
servediff diff old.json new.json
```

默认只显示和推理相关的包、GPU、引擎环境变量；其他包要加 `--all` 才列出。名字里含 `TOKEN`、`KEY`、`SECRET`、`PASSWORD` 等的环境变量，值一律替换成 `<redacted>`。快照和日志里有本机路径，分享之前请先检查。

## 边界

- 已测试：vllm#47047、#53472 里贴出的日志片段（v0.23、v0.26），vLLM 0.28.0 / 0.30.0 的完整启动日志，以及一份 SGLang 日志。都是单卡、dense 模型。
- TP>1、完整日志里的 MoE 模型、其他版本，都还没有验证。
- 识别不到的行会显示为 `<absent>`，不会显示成错误的值。

许可证：Apache-2.0。
