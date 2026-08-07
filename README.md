# DR-Tulu-8B on BrowseComp-Plus — self-contained bundle

Runs DR-Tulu-8B (Base / SFT / RL) on all 830 BrowseComp-Plus questions with a
BM25 retriever, producing the trajectories needed for Table 1.

Everything is pre-configured and verified. **Run the four steps in order.**
Pull fixes any time with `git pull` -- no re-setup needed.

```bash
git clone https://github.com/riteshbhirud/drtulu.git && cd drtulu

bash 1_setup.sh                    # env + data + models (~1-2 h, ~120 GB)
bash 2_verify.sh                   # MUST print ALL CHECKS PASSED

bash 3_run.sh rl --limit 5         # smoke test (~15 min)
bash 4_check.sh rl                 # MUST print HEALTHY

bash 3_run.sh rl                   # full 830 questions
bash 4_check.sh rl                 # check again when done

bash 3_run.sh sft   &&  bash 4_check.sh sft
bash 3_run.sh base  &&  bash 4_check.sh base
```

## Shared-node GPU selection (automatic)

`3_run.sh` picks GPUs at launch time, so it adapts to whatever else is running:

- a GPU counts as usable only if it has **>= 12 GiB free** (`MIN_FREE_GIB`)
- it takes **at most 4** (`MAX_GPUS`) so we never hog a shared box
- the count is rounded down to a valid tensor-parallel size (1, 2, or 4 --
  DR-Tulu has 8 KV heads, so 3 GPUs is not a legal configuration)
- it prefers the emptiest GPUs, sitting beside other jobs rather than on them
- if nothing is free it stops with a clear message instead of OOM-ing

Override when you want to:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash 3_run.sh rl   # force specific GPUs
MAX_GPUS=2 bash 3_run.sh rl                     # be even politer
MIN_FREE_GIB=10 bash 3_run.sh rl                # accept tighter cards
GPU_UTIL=0.70 bash 3_run.sh rl                  # if vLLM OOMs beside other jobs
```

VRAM needed: weights are 15.3 GiB (bf16), sharded across GPUs -- 3.8 GiB/GPU at
TP=4. Each 40,960-token sequence costs 5.6 GiB of KV cache, also sharded.
On 22-27 GiB free per card, TP=4 gives roughly 12-15 concurrent sequences.

## Requirements

- **Python >= 3.12** — the scaffold's browser tool imports `gpt_oss`, and every
  `gpt-oss` release on PyPI requires-python >= 3.12. If the system python is
  older, `1_setup.sh` installs a local CPython 3.12 via `uv` (no root).
  An `env/` built on an older python is detected and rebuilt automatically.
- **No root needed.** If `python3 -m venv` fails because
  `ensurepip` is missing (common on Debian/Ubuntu clusters), `1_setup.sh`
  automatically falls back to: venv+get-pip, then `virtualenv`, then conda.
- **Java 21** — pyserini/Lucene cannot open the index on Java 11. `1_setup.sh`
  downloads a Temurin JDK 21 into `jdk21/` automatically when the system Java is
  older, and writes `env.sh` so the other scripts pick it up. No root required.
- 1-8 NVIDIA GPUs. DR-Tulu-8B is ~16 GB in bf16, so **a 40 GB card is enough**;
  more GPUs mainly buy concurrency.
- ~120 GB disk (48 GB models + 6 GB data + env)
- Internet for step 1 only. Steps 2-4 run offline.

## Configuration (already set — do not change casually)

| Setting | Value | Why |
|---|---|---|
| Turn cap | **60** | Mentor's instruction; matches the TRACE paper's rollout protocol |
| Context | **40,960** | DR-Tulu's published eval setting (their repo's `vllm serve` line) |
| Retriever | **BM25 top-5** | Official BrowseComp-Plus protocol |
| Snippet cap | 2048 chars | ~512 tokens, the official BCP cap |
| Temperature | **1.0** | DR-Tulu's published eval config |
| Prompt | mentor's `Dr_tUlU.py`, verbatim | uniform tool interface across systems |
| Tools | `search` / `open` / `find` | the mentor's adapted interface |

## Resuming — you cannot lose work

`3_run.sh` scans what is already finished and skips it. Kill it at any point
(Ctrl-C, node reboot, preemption) and re-run the **same command**; it continues
from where it stopped. At most one in-flight question is redone. This was
verified across 7 preemption scenarios including a kill mid-write.

## The one failure mode to watch for

DR-Tulu's prompt tells the model to emit XML tool calls:

```
<call_tool name="search" topn="10">query</call_tool>
```

The stock OpenResearcher scaffold only understands `<tool_call>{json}</tool_call>`.
**This bundle ships a patched scaffold with a `<call_tool>` parser.** Without it
the failure is completely silent: every tool call is dropped, the agent loops to
the turn cap, and you get 830 clean-looking trajectories at 0% success with no
error message anywhere.

`2_verify.sh` step 5 and `4_check.sh` both test for this. **If either fails, stop
and report it — do not run 830 questions.**

## What to send back

When a checkpoint reaches 830/830 and `4_check.sh` says HEALTHY:

```
results/drtulu_<ckpt>_bm25_top5/     # the trajectories
logs/drtulu_<ckpt>_bm25_top5_*.log   # run logs
```

Judging (Qwen3-32B with the official BC-Plus grader) runs back on the UMass
side, so these files are all that is needed.

## If something goes wrong

- **`2_verify.sh` fails step 3** — Java is missing or not 21. pyserini cannot
  open the Lucene index without it.
- **vLLM fails to start** — try `GPU_UTIL=0.70 bash 3_run.sh rl` on a shared
  node, or set `CUDA_VISIBLE_DEVICES` to GPUs that are actually free.
- **Many `status=fail` records** — vLLM died while the agent kept running. Those
  records are counted as "done" and would be skipped forever on resume. Delete
  them from the shard file before resuming.
- **`4_check.sh` says NO TOOL CALLS** — the parser is not working. Do not
  continue; report it.
