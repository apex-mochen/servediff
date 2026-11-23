# The log you pasted is not the log the server wrote

Two pastes, two different losses, same cause: a forum or a chat client rewrites punctuation,
and every rule in this tool is written against the ASCII the server prints.

## Loss 1 — quotes (vLLM 0.28, forum 2946)

```
Mamba cache mode is set to ‘align’ for Qwen3_5ForConditionalGeneration …
min_p and logit_bias parameters won’t work with speculative decoding.
```

The server writes `'align'` and `won't`. The rule wants `'align'`. Result: `kv_cache.mamba_mode`
and the spec warning both disappear from a log that plainly contains them — the worst kind of
failure, because the table still looks complete.

## Loss 2 — dashes (vLLM 0.19, forum 2570)

Hugging Face puts **two hyphens** in a cache directory name:

```
/root/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/61a5771f…
```

SmartyPants turns `--` into an en dash, so the pasted copy reads:

```
/root/.cache/huggingface/hub/models–Qwen–Qwen3.6-35B-A3B-FP8/snapshots/61a5771f…
```

`arg.model` is still read — but with the wrong value. Two consequences worth separating:

- `diff` between a pasted log and a log taken on the box reports a **model path change that
  never happened**, and the real change next to it is easier to miss.
- Anything downstream that keys off the model path (a cache, a baseline file, a group-by in
  `show`) sees a second identity for one checkpoint.

`---` becomes an em dash, so the fold has to be `– → --` and `— → ---`, not "drop the dash".

## The fix

One translation table, applied in `parse` and in `audit` before any rule runs:

```python
CURLY = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                       "\u2013": "--", "\u2014": "---"})
```

## What this says about pasted logs generally

The forum is a lossy channel, and the loss is **typographic, silent and content-shaped**:
it hits paths, flags and quoted values, never the line structure. So the cheap defence is
one fold of the characters that have a known ASCII twin, applied before matching rather than
inside each rule. Anything the fold cannot know — a wrapped line, a truncated paste, a log
reassembled from screenshots — is a different problem and is not solved by this.

This is also why the corpus keeps the pasted punctuation in the fixture and folds it at read
time: the fixture is the evidence of what a user actually has, and the fold is the product.
