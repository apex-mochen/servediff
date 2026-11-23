# Your PowerShell redirect wrote UTF-16, and every log tool read it as mojibake

You captured a server log on Windows the natural way:

```powershell
vllm serve Qwen/Qwen3-4B 2>&1 > serve.log
```

Then you opened it and told yourself the log is fine — it is, in Notepad. Then you fed it to
`servediff show serve.log`, or to `grep`, or to any of the log-upload forms, and got back:

```
serve.log: no vLLM or SGLang configuration lines recognised
```

The log is not the problem. The file is UTF‑16LE, and you are not.

## The one line that costs an hour

`>` and `Out-File` are not byte copy. In PowerShell 5.1 — still the default `powershell.exe` on
every Windows machine that has not been told otherwise — both write **UTF‑16LE with a BOM**. The
redirect encoding is `$OutputEncoding`'s cousin, not the console's code page, so a server that
prints clean UTF‑8 to the terminal ends up in a file where every ASCII byte is followed by a
`0x00`.

Read that as UTF‑8 and nothing fails loudly: the decoder produces `\x00` and replacement
characters, every `[module.py:851]` anchor is broken, and a parser looking for
`[kv_cache_utils.py:851]` finds a line of NULs. You conclude the log has no configuration in it,
which is exactly wrong.

```powershell
PS> $b = [IO.File]::ReadAllBytes("serve.log")
PS> $b[0..5] -join ','
255,254,73,0,78,0
```

`FF FE` is the UTF‑16LE BOM, and `73,0,78,0` is `I`, NUL, `N`, NUL. That is all the diagnosis
there is.

## Four ways out, in the order I would try them

**1. Let the tool sniff the bytes.** Since 0.4.74 `servediff` reads the BOM (and a NUL-density
heuristic for files written without one) and decodes accordingly, saying so on stderr, so stdout
stays pipeable:

```
note: serve.log was read as utf-16, not UTF-8
```

That covers `show`, `diff`, `audit`, `repeats`, and stdin (`ssh node 'cat serve.log' | servediff
show -`) — a UTF‑16 paste over the wire is decoded the same way.

**2. Convert once, then use everything else.** `grep`, `awk`, `jq` and every upload form
downstream also want UTF‑8:

```powershell
(Get-Content serve.log) | Set-Content -Encoding utf8 serve-utf8.log
```

The `(Get-Content ...)` wrapper matters: `Get-Content` is slow but decodes correctly, and
`Set-Content` then writes the bytes you expect. Add `-ReadCount 0` on a 200 MB log to keep it to
one pass.

**3. Redirect through `cmd` instead of PowerShell.** `cmd.exe`'s `>` is a plain byte copy:

```powershell
cmd /c "vllm serve Qwen/Qwen3-4B > serve.log 2>&1"
```

**4. Change the session default.** `$OutputEncoding = [Text.UTF8Encoding]::new($false)` and
`[Console]::OutputEncoding` both aim at the pipeline, not at `>`; the reliable lever is
`Out-File -Encoding utf8`, which in 5.1 writes UTF‑8 **with** a BOM — harmless to every reader
here, but worth knowing before you diff two captures byte-for-byte.

## Why this matters more than it looks

A log that reads as empty fails silently and blames the wrong thing: you go looking for a
missing `--shm-size`, a wrong `CUDA_VISIBLE_DEVICES`, a build that printed nothing, when the
capture never made it to disk as text. It also breaks the *comparison*, which is the entire point
of keeping a log: two captures on the same box in different encodings will differ in ways no
reader can attribute to the server.

So: check `FF FE` first, or let the tool check it for you, and then argue about the server.

`servediff show` prints facts from a vLLM / SGLang startup log — the resolved backends, CUDA
graph sizes, KV cache, the failure at the end — and `servediff diff a.log b.log` says which
facts moved between two runs. `pip install servediff`. Source and notes:
<https://github.com/apex-mochen/servediff>.
