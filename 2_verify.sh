#!/bin/bash
# DR-Tulu bundle — STEP 2: verify everything BEFORE spending GPU hours.
#
#   bash 2_verify.sh
#
# Every check here has caught a real failure during the UMass runs. The most
# important is the <call_tool> parser: DR-Tulu's prompt emits XML tool calls
# that the stock scaffold cannot parse, and the failure is SILENT -- you get
# 830 well-formed trajectories at 0% success with no error anywhere.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/env/bin/activate"
[ -f "$ROOT/env.sh" ] && source "$ROOT/env.sh"
export JAVA_HOME="${JAVA_HOME:-}"
FAIL=0
ok(){ echo "  [ OK ] $1"; }
bad(){ echo "  [FAIL] $1"; FAIL=1; }

echo "=========================================================="
echo "DR-Tulu bundle verification"
echo "=========================================================="

echo "[0] python >= 3.12 (gpt-oss has no build below this)"
python -c 'import sys;sys.exit(0 if sys.version_info>=(3,12) else 1)' \
  && ok "python $(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')" \
  || bad "python $(python -c 'import sys;print("%d.%d"%sys.version_info[:2])') -- need >=3.12; delete env/ and re-run 1_setup.sh"

echo "[1] python imports"
python - <<'PY' && ok "vllm/torch/pyserini/json5/tevatron import" || bad "core imports"
import vllm, torch, json5, faiss, peft
from pyserini.search.lucene import LuceneSearcher
from tevatron.retriever.arguments import ModelArguments
PY

echo "[2] GPU visible to torch"
python - <<'PY' && ok "cuda available" || bad "cuda NOT available"
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
python -c "import torch;print(f'       {torch.cuda.device_count()} GPU(s), {torch.cuda.get_device_name(0)}')" 2>/dev/null || true

echo "[2b] Java 21 (pyserini/Lucene cannot open the index on Java 11)"
if java -version 2>&1 | grep -qE '"(21|22|23)'; then
  ok "java $(java -version 2>&1 | head -1 | grep -oE '"[^"]+"')"
else
  bad "java is $( (java -version 2>&1|head -1) || echo missing) -- need 21. Re-run 1_setup.sh, or: conda install -c conda-forge openjdk=21"
fi

echo "[3] BM25 index — corpus size and known-answer query"
python - <<'PY' && ok "BM25 index correct (100195 docs, rank-1 docid 5412)" || bad "BM25 index wrong/unreadable"
import sys
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("data/browsecomp-plus-indexes/bm25")
n = s.num_docs
h = s.search("Queen Arwa University cultural activities 2002 Yemen", 5)
print(f"       num_docs={n}  rank1={h[0].docid if h else 'NONE'}")
sys.exit(0 if (n == 100195 and h and h[0].docid == "5412") else 1)
PY

echo "[4] dataset decrypts to 830 questions"
python - <<'PY' && ok "830 questions load" || bad "dataset load"
import sys
sys.path.insert(0, "scaffold/OpenResearcher")
from data_utils import load_dataset
d = load_dataset("browsecomp_plus", data_path="data/browsecomp-plus/data/*.parquet")
print(f"       {len(d)} questions; first qid={d[0]['qid']}")
sys.exit(0 if len(d) == 830 else 1)
PY

echo "[5] <call_tool> XML parser — THE critical DR-Tulu check"
python - <<'PY' && ok "call_tool parser round-trips all 3 tool forms" || bad "call_tool parser BROKEN -- DO NOT RUN"
import sys
sys.path.insert(0, "scaffold/OpenResearcher")
from deploy_agent import parse_call_tool_xml
cases = [
    ('<call_tool name="search" topn="10">my query</call_tool>', "search", "my query"),
    ('<call_tool name="open" cursor="0" id="3"></call_tool>',    "open",   None),
    ('<call_tool name="find" cursor="1">exact text</call_tool>', "find",   "exact text"),
    # DR-Tulu emits its TRAINING tool names (google_search / browse_webpage /
    # snippet_search) even though the mentor prompt teaches search/open/find.
    # Observed live on ORNL: every call was google_search and all were dropped.
    ('<call_tool name="google_search" topn="10">q</call_tool>',   "search", "q"),
    ('<call_tool name="browse_webpage" id="3"></call_tool>',      "open",   None),
    ('<call_tool name="snippet_search" topn="5">q</call_tool>',   "search", "q"),
]
for raw, want, txt in cases:
    got = parse_call_tool_xml(raw)
    if not got or got.get("name") != want:
        print(f"       FAILED on {raw}"); sys.exit(1)
    if txt and txt not in str(got.get("arguments", {})):
        print(f"       arg lost on {raw}"); sys.exit(1)
    print(f"       {want:<7} -> {got['arguments']}")
sys.exit(0)
PY

echo "[5b] rendered prompt must NOT contain a conflicting tool format"
python - <<'PY2' && ok "rendered prompt teaches call_tool only" || bad "chat template injects <tool_call> -- model will ignore the mentor prompt"
import os, sys, json
sys.path.insert(0, "scaffold/OpenResearcher")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/DR-Tulu-8B", trust_remote_code=True)
prompt = open("prompts/drtulu_prompt.txt").read()
msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": "Q?"}]
# 3_run.sh sets RUN_TOOL_FORMAT=call_tool_xml, which makes deploy_agent pass
# tools=None. Reproduce that here.
r = tok.apply_chat_template(msgs, tools=None, tokenize=False, add_generation_prompt=True)
n_ct, n_tc = r.count("call_tool"), r.count("<tool_call>")
print(f"       call_tool={n_ct}  <tool_call>={n_tc}")
if n_ct < 5:
    print("       mentor prompt not reaching the model"); sys.exit(1)
if n_tc > 0:
    print("       template injected a CONFLICTING <tool_call> format"); sys.exit(1)
sys.exit(0)
PY2

echo "[6] malformed-tool-call salvage"
python - <<'PY' && ok "salvage recovers malformed calls, rejects garbage" || bad "salvage"
import sys
sys.path.insert(0, "scaffold/OpenResearcher")
from deploy_agent import salvage_tool_call
good = salvage_tool_call(r'{"name":"browser.search","arguments":{"query": \"a b c\", "topn": 5}}')
assert good and good["arguments"].get("query") == "a b c", good
for junk in ["", "prose", '{"foo":1}', '{"name":"evil.exec","arguments":{"query":"x"}}']:
    assert salvage_tool_call(junk) is None, junk
print("       repair OK, garbage rejected")
PY

echo "[7] config: 60-turn cap, top-5, 40960 context"
grep -q 'MAX_ROUNDS:-60' 3_run.sh && ok "MAX_ROUNDS=60" || bad "MAX_ROUNDS not 60"
grep -q 'RUN_TOPK="5"' 3_run.sh && ok "top_k=5" || bad "top_k not 5"
grep -q 'RUN_CTX="40960"' 3_run.sh && ok "context=40960 (DR-Tulu published eval)" || bad "context not 40960"

echo "[8] mentor's DR-Tulu prompt present and intact"
if grep -q 'call_tool name="search"' prompts/drtulu_prompt.txt \
   && grep -q "BrowseComp Plus corpus" prompts/drtulu_prompt.txt; then
  ok "prompt verbatim ($(wc -c < prompts/drtulu_prompt.txt) bytes)"
else
  bad "prompt missing/modified"
fi

echo "[9] models present"
for m in DR-Tulu-8B DR-Tulu-SFT-8B Qwen3-8B; do
  if [ -f "models/$m/config.json" ]; then ok "models/$m"; else bad "models/$m missing"; fi
done

echo "=========================================================="
if [ "$FAIL" -eq 0 ]; then
  echo "ALL CHECKS PASSED — safe to run."
  echo "NEXT:  bash 3_run.sh rl --limit 5     # 5-question smoke test FIRST"
else
  echo "SOME CHECKS FAILED — fix before running, or you will burn GPU hours"
  echo "producing data that looks fine and is worthless."
  exit 1
fi
echo "=========================================================="
