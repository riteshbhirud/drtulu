#!/bin/bash
# DR-Tulu bundle — STEP 4: check the data is real, not silently broken.
#
#   bash 4_check.sh rl        # after the smoke test AND after the full run
#
# Low success rate is NOT a failure here -- BrowseComp-Plus is hard, and the
# reference number for a Qwen3-4B-class base model is ~7%. These checks look
# for the signatures of a BROKEN HARNESS, which is a different thing.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/env/bin/activate"
CKPT="${1:-rl}"
RUN="drtulu_${CKPT}_bm25_top5"

python - "$ROOT/results/$RUN" <<'PY'
import json, glob, os, sys, collections
d = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(d, "node_*_shard_*.jsonl")):
    for line in open(f, encoding="utf-8"):
        try: rows.append(json.loads(line))
        except Exception: pass       # torn line from a kill; resume will redo it
if not rows:
    print("NO RECORDS in", d); sys.exit(1)

n = len(rows)
fails = sum(1 for r in rows if r.get("status") == "fail")
reasons = collections.Counter(r.get("terminated_reason") for r in rows)
tools = collections.Counter()
att = rep = drop = 0
zero_tool = 0
for r in rows:
    c = 0
    for m in r.get("messages", []):
        if m.get("role") != "assistant": continue
        for tc in (m.get("tool_calls") or []):
            c += 1
            tools[tc["function"]["name"].split(".")[-1]] += 1
    if c == 0: zero_tool += 1
    att  += r.get("tool_calls_attempted", 0)
    rep  += r.get("tool_calls_repaired", 0)
    drop += r.get("tool_calls_dropped", 0)
rounds = [r.get("rounds_used", 0) for r in rows]

print("=" * 58)
print(f"DR-Tulu {os.path.basename(d)}   {n}/830 records")
print("=" * 58)
print(f"  tools used            {dict(tools)}")
print(f"  terminated            {dict(reasons)}")
print(f"  mean rounds           {sum(rounds)/n:.1f}   (cap 60)")
print(f"  tool calls            {att} attempted, {rep} repaired, {drop} dropped")
print()

bad = []
# THE critical one: DR-Tulu emits <call_tool> XML. If the parser is not working
# you get zero tool calls and a run that looks fine but is worthless.
if not tools:
    bad.append("NO TOOL CALLS AT ALL -> <call_tool> parser is not working. "
               "Data is worthless. Re-run 2_verify.sh.")
if zero_tool / n > 0.20:
    bad.append(f"{zero_tool}/{n} questions made no tool call at all")
if fails / n > 0.05:
    bad.append(f"{fails}/{n} status=fail -> vLLM likely died mid-run; "
               "delete those records before resuming or they are skipped forever")
if att and drop / att > 0.25:
    bad.append(f"{drop/att:.0%} of tool calls unrecoverable")
if reasons.get("answer", 0) / n < 0.05:
    bad.append("almost nothing terminated with an <answer> tag")

if bad:
    print("SUSPICIOUS — investigate before trusting these numbers:")
    for b in bad: print("   -", b)
    print("\n  (Low SR alone is expected and fine. These flags mean the HARNESS")
    print("   may be broken, which is a different problem.)")
    sys.exit(1)
print("HEALTHY — the agent is exercising the tools correctly.")
print("Whatever success rate this run produces is a valid measurement.")
sys.exit(0)
PY
rc=$?
echo
if [ $rc -eq 0 ]; then
  echo "Next: continue running, or when 830/830 is done, send back:"
  echo "   results/$RUN/   and   logs/${RUN}_*.log"
fi
exit $rc
