# Where this corpus came from

These 28 logs were collected on 2026-11-10 with the same command the earlier
directories used:

```
GET /search/issues?q=repo:<repo> is:issue created:>2026-11-05&per_page=40
  -> take the longest fenced block in the body
  -> keep it only if it has 40 source lines or more
```

Both repositories were queried; 40 issues per repository were read (vLLM reported 113
matches, SGLang 75 in that window). Everything with a long block was written out
verbatim, one file per issue, 3306 lines in total.

Four files also appear in `held-out-2026-10` (`sglang-49123`, `49129`, `49132`, `49133`).
That directory was collected by date-of-collection, not by issue date, so its window
overlapped this one; the duplicates are kept so each directory is a single command's
output and can be re-run on its own.

## Files

| file | issue | created | lines |
|---|---|---|---|
| `sglang-48802.log` | sgl-project/sglang#48802 | 2026-11-06T04:26:08Z | 79 |
| `sglang-48904.log` | sgl-project/sglang#48904 | 2026-11-06T13:02:00Z | 46 |
| `sglang-48906.log` | sgl-project/sglang#48906 | 2026-11-06T13:03:07Z | 72 |
| `sglang-48911.log` | sgl-project/sglang#48911 | 2026-11-06T14:00:29Z | 49 |
| `sglang-48913.log` | sgl-project/sglang#48913 | 2026-11-06T14:00:37Z | 53 |
| `sglang-48917.log` | sgl-project/sglang#48917 | 2026-11-06T14:05:02Z | 59 |
| `sglang-48919.log` | sgl-project/sglang#48919 | 2026-11-06T14:10:41Z | 49 |
| `sglang-48930.log` | sgl-project/sglang#48930 | 2026-11-06T16:55:32Z | 84 |
| `sglang-49123.log` | sgl-project/sglang#49123 | 2026-11-08T03:00:19Z | 195 |
| `sglang-49129.log` | sgl-project/sglang#49129 | 2026-11-08T04:47:00Z | 46 |
| `sglang-49132.log` | sgl-project/sglang#49132 | 2026-11-08T05:33:44Z | 48 |
| `sglang-49133.log` | sgl-project/sglang#49133 | 2026-11-08T05:34:09Z | 48 |
| `vllm-66074.log` | vllm-project/vllm#66074 | 2026-11-06T08:16:15Z | 83 |
| `vllm-66109.log` | vllm-project/vllm#66109 | 2026-11-06T12:16:52Z | 225 |
| `vllm-66113.log` | vllm-project/vllm#66113 | 2026-11-06T12:49:59Z | 41 |
| `vllm-66115.log` | vllm-project/vllm#66115 | 2026-11-06T13:14:01Z | 59 |
| `vllm-66120.log` | vllm-project/vllm#66120 | 2026-11-06T13:44:14Z | 62 |
| `vllm-66124.log` | vllm-project/vllm#66124 | 2026-11-06T14:09:10Z | 73 |
| `vllm-66129.log` | vllm-project/vllm#66129 | 2026-11-06T15:22:11Z | 167 |
| `vllm-66192.log` | vllm-project/vllm#66192 | 2026-11-07T00:00:57Z | 230 |
| `vllm-66193.log` | vllm-project/vllm#66193 | 2026-11-07T00:04:06Z | 197 |
| `vllm-66195.log` | vllm-project/vllm#66195 | 2026-11-07T00:09:58Z | 242 |
| `vllm-66211.log` | vllm-project/vllm#66211 | 2026-11-07T02:15:58Z | 70 |
| `vllm-66241.log` | vllm-project/vllm#66241 | 2026-11-07T07:20:53Z | 234 |
| `vllm-66242.log` | vllm-project/vllm#66242 | 2026-11-07T07:27:01Z | 234 |
| `vllm-66244.log` | vllm-project/vllm#66244 | 2026-11-07T07:34:17Z | 234 |
| `vllm-66246.log` | vllm-project/vllm#66246 | 2026-11-07T07:57:21Z | 93 |
| `vllm-66248.log` | vllm-project/vllm#66248 | 2026-11-07T07:58:39Z | 234 |

## What this corpus is not

It is not a random sample of log space, and it cannot show that the noise table covers
every shape. It is one month's worth of what people pasted into these two repositories,
read with the rules that existed before it arrived — which is the only property that
matters for the test that uses it.
