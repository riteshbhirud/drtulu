#!/bin/bash
# Dump exactly what the model SAW and what it EMITTED, from a completed run.
# Use when 4_check.sh reports zero tool calls: it distinguishes
#   (a) the prompt never reached the model,
#   (b) the model emitted call_tool but the parser missed it,
#   (c) the model ignored the format and answered in prose.
#
#   bash 5_debug_prompt.sh rl
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/env/bin/activate"
CKPT="${1:-rl}"
RUN="drtulu_${CKPT}_bm25_top5"

python - "$ROOT/results/$RUN" <<'PY'
import json, glob, os, sys, re
d = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(d, "node_*_shard_*.jsonl")):
    for line in open(f, encoding="utf-8"):
        try: rows.append(json.loads(line))
        except Exception: pass
if not rows:
    print("NO RECORDS"); sys.exit(1)
r = rows[0]
msgs = r.get("messages", [])

print("=" * 70)
print("1. SYSTEM PROMPT AS STORED  (what run_one built)")
print("=" * 70)
sysmsg = msgs[0]["content"] if msgs and msgs[0].get("role") == "system" else "(none)"
print(f"  length: {len(sysmsg)} chars")
print(f"  mentions 'call_tool' : {sysmsg.count('call_tool')}")
print(f"  mentions '<tool_call>': {sysmsg.count('<tool_call>')}")
print("  --- first 300 ---")
print("   ", repr(sysmsg[:300]))
print("  --- last 300 ---")
print("   ", repr(sysmsg[-300:]))

print()
print("=" * 70)
print("2. WHAT THE MODEL EMITTED  (assistant turn 1, raw)")
print("=" * 70)
for m in msgs:
    if m.get("role") != "assistant":
        continue
    c = m.get("content") or ""
    rc = m.get("reasoning_content") or ""
    print(f"  content length: {len(c)}   reasoning length: {len(rc)}")
    print(f"  contains '<call_tool'  : {'<call_tool' in c or '<call_tool' in rc}")
    print(f"  contains '<tool_call>' : {'<tool_call>' in c or '<tool_call>' in rc}")
    print(f"  contains '<answer>'    : {'<answer>' in c}")
    print(f"  parsed tool_calls      : {m.get('tool_calls')}")
    print("  --- reasoning (first 500) ---")
    print("   ", repr(rc[:500]))
    print("  --- content (first 900) ---")
    print("   ", repr(c[:900]))
    break

print()
print("=" * 70)
print("3. DIAGNOSIS")
print("=" * 70)
a = next((m for m in msgs if m.get("role") == "assistant"), {})
blob = (a.get("content") or "") + (a.get("reasoning_content") or "")
if sysmsg.count("call_tool") < 5:
    print("  -> The mentor prompt is NOT reaching the model.")
    print("     SYSTEM_PROMPT_FILE is wrong, or run_one overrode it.")
elif "<call_tool" in blob:
    print("  -> Model DID emit <call_tool>; the PARSER missed it.")
    print("     Paste section 2 -- the parser regex needs widening.")
elif "<tool_call>" in blob:
    print("  -> Model emitted <tool_call> JSON instead of <call_tool> XML.")
    print("     A conflicting tool spec is still reaching it.")
else:
    print("  -> Model emitted NEITHER format; it answered in prose.")
    print("     It is ignoring the tool instructions entirely. Likely causes:")
    print("       * thinking mode consumed the turn and it jumped to <answer>")
    print("       * the prompt is present but the model was not trained on it")
    print("     Paste section 2 so the actual text can be read.")
PY
