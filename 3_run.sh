#!/bin/bash
# DR-Tulu bundle — STEP 3: run BrowseComp-Plus.
#
#   bash 3_run.sh rl   --limit 5   # smoke test FIRST (~15 min)
#   bash 3_run.sh rl               # full 830 questions
#   bash 3_run.sh sft
#   bash 3_run.sh base
#
# RESUMABLE: re-running continues where it stopped. Nothing is lost or repeated.
# Kill it any time (Ctrl-C, node reboot, preemption) and re-run the same command.
#
# GPUs: set CUDA_VISIBLE_DEVICES to the free ones, e.g.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash 3_run.sh rl
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CKPT="${1:-rl}"; shift || true
case "$CKPT" in
  rl)   MODEL="models/DR-Tulu-8B" ;;
  sft)  MODEL="models/DR-Tulu-SFT-8B" ;;
  base) MODEL="models/Qwen3-8B" ;;
  *) echo "usage: bash 3_run.sh {rl|sft|base} [--limit N]"; exit 1 ;;
esac

RUN_NAME="drtulu_${CKPT}_bm25_top5"
OUT="$ROOT/results/$RUN_NAME"
mkdir -p "$OUT" "$ROOT/logs" "$ROOT/checkpoints"

source "$ROOT/env/bin/activate"
[ -f "$ROOT/env.sh" ] && source "$ROOT/env.sh"
export JAVA_HOME="${JAVA_HOME:-}"

# ---- CONFIG (verified against the mentor's spec; do not change casually) ----
export RUN_CTX="40960"          # DR-Tulu's published eval context
export MAX_ROUNDS="${MAX_ROUNDS:-60}"   # mentor's turn cap (TRACE paper)
export RUN_TOPK="5"             # BM25 top-5, official BCP protocol
export GEN_TEMPERATURE="1.0"    # DR-Tulu eval config (auto_search_sft.yaml)
export MODEL_CONTEXT_WINDOW="$RUN_CTX"
export SYSTEM_PROMPT_FILE="$ROOT/prompts/drtulu_prompt.txt"
export CHECKPOINT_FILE="$ROOT/checkpoints/${RUN_NAME}.json"
export TOKENIZER_PATH="$ROOT/$MODEL"
export RUN_SYSTEM="DR-Tulu-8B"
export RUN_CHECKPOINT="$CKPT"
export RUN_RETRIEVER="bm25"
export RUN_TOOL_FORMAT="call_tool_xml"
export RUN_THINKING="true"
export VLLM_USE_FLASHINFER_SAMPLER=0    # FlashInfer JIT fails on some clusters
# vLLM's torch.compile path needs nvcc + libnvrtc. Clusters often ship only the
# driver, but the CUDA toolkit is bundled inside the venv's nvidia/ packages --
# point CUDA_HOME at it. Cost us two failed starts on the UMass box.
for _cu in "$ROOT/env/lib/python3.12/site-packages/nvidia/cu13" \
           "$ROOT/env/lib/python3.12/site-packages/nvidia/cu12"; do
  if [ -x "$_cu/bin/nvcc" ]; then
    export CUDA_HOME="$_cu"
    export PATH="$_cu/bin:$PATH"
    export LD_LIBRARY_PATH="$_cu/lib:${LD_LIBRARY_PATH:-}"
    echo "[cuda] CUDA_HOME=$_cu"
    break
  fi
done
if [ -z "${CUDA_HOME:-}" ] && ! command -v nvcc >/dev/null 2>&1; then
  # No toolkit anywhere: disable the compile path so vLLM falls back to eager.
  export VLLM_USE_V1=1
  export TORCHINDUCTOR_DISABLE=1
  echo "[cuda] no nvcc found -- disabling inductor compile path"
fi
export HF_HUB_OFFLINE=1                 # weights are local; never hit the network
export TMPDIR="${TMPDIR:-/tmp}"
# On a SHARED node, GPU peer-to-peer and NCCL shared-memory transports often
# fail between devices that belong to different jobs -- vLLM's TP workers then
# die inside init_device(). Force the portable transports.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Shared node: pick GPUs that are actually free RIGHT NOW, capped so we never
# take the whole box. Respects an explicit CUDA_VISIBLE_DEVICES if you set one.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  DEVS=$(bash "$ROOT/pick_gpus.sh") || {
    echo "FATAL: no usable GPUs right now (see messages above). Try later, or"
    echo "       MIN_FREE_GIB=10 bash 3_run.sh $CKPT"
    exit 1
  }
  export CUDA_VISIBLE_DEVICES="$DEVS"
else
  echo "[gpu] using caller-supplied CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
# vLLM tensor-parallel must divide the model's 8 KV heads: 1,2,4,8 only.
case "$NGPU" in 8) TP=8;; 4|5|6|7) TP=4;; 2|3) TP=2;; *) TP=1;; esac
CONC=$(( TP * 16 ))
SEARCH_PORT=$(( 18000 + RANDOM % 900 ))
VLLM_PORT=$(( 19000 + RANDOM % 900 ))

# Fail fast with an actionable message rather than a 60-line duckdb/uvicorn
# traceback 30 seconds into service startup.
for g in "$ROOT/data/browsecomp-plus/data/*.parquet" \
         "$ROOT/data/browsecomp-plus-corpus/data/*.parquet" \
         "$ROOT/data/browsecomp-plus-indexes/bm25/*" \
         "$ROOT/$MODEL/config.json"; do
  compgen -G "$g" > /dev/null 2>&1 || {
    echo "FATAL: missing required files: $g"
    echo "   Setup did not finish. Run:  bash 1_setup.sh  (it resumes)"
    echo "   Then:                       bash 2_verify.sh"
    exit 1
  }
done

DONE_N=$(cat "$OUT"/node_*_shard_*.jsonl 2>/dev/null | wc -l)
echo "=========================================================="
echo "DR-Tulu $CKPT  |  $MODEL"
echo "  ctx=$RUN_CTX  turns=$MAX_ROUNDS  top_k=$RUN_TOPK  temp=$GEN_TEMPERATURE"
echo "  GPUs visible=$NGPU -> TP=$TP, concurrency=$CONC"
echo "  already done: $DONE_N/830   (resuming)"
echo "  ports: search=$SEARCH_PORT vllm=$VLLM_PORT"
echo "=========================================================="

cleanup() {
  echo; echo "[cleanup] stopping services"
  # ORDER MATTERS: agent first. Killing vLLM under a live agent makes every
  # in-flight question fail its retries and be written as an error record --
  # which resume then treats as "done" and skips forever.
  [ -n "${AGENT_PID:-}" ] && { kill "$AGENT_PID" 2>/dev/null; wait "$AGENT_PID" 2>/dev/null; }
  [ -n "${VLLM_PID:-}"  ] && kill "$VLLM_PID" 2>/dev/null
  [ -n "${SEARCH_PID:-}" ] && kill "$SEARCH_PID" 2>/dev/null
  sleep 4
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null   # holds GPU memory after the API server exits
}
trap cleanup EXIT TERM INT

# ---------------------------------------------------------------- BM25 service
cd "$ROOT/scaffold/OpenResearcher"
LUCENE_EXTRA_DIR="$PWD/lucene_jars" \
LUCENE_INDEX_DIR="$ROOT/data/browsecomp-plus-indexes/bm25" \
CORPUS_PARQUET_PATH="$ROOT/data/browsecomp-plus-corpus/data/*.parquet" \
SEARCHER_TYPE=bm25 MAX_SNIPPET_LEN=2048 \
  python -m uvicorn scripts.deploy_search_service:app \
    --host 127.0.0.1 --port "$SEARCH_PORT" > "$ROOT/logs/${RUN_NAME}_search.log" 2>&1 &
SEARCH_PID=$!
cd "$ROOT"

# ---------------------------------------------------------------- vLLM
python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$VLLM_PORT" \
  --max-model-len "$RUN_CTX" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --tensor-parallel-size "$TP" \
  --max-num-seqs "$CONC" \
  --distributed-executor-backend mp \
  --disable-custom-all-reduce \
  --trust-remote-code --enable-prefix-caching \
  > "$ROOT/logs/${RUN_NAME}_vllm.log" 2>&1 &
VLLM_PID=$!

echo "[wait] services starting (vLLM load + CUDA graphs can take 3-8 min)..."
for i in $(seq 1 150); do
  s=0; v=0
  curl -s -m 2 "http://127.0.0.1:$SEARCH_PORT/" >/dev/null 2>&1 && s=1
  curl -s -m 2 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1 && v=1
  [ $s -eq 1 ] && [ $v -eq 1 ] && { echo "[wait] both up (~$((i*10))s)"; break; }
  kill -0 "$VLLM_PID" 2>/dev/null || {
    echo "FATAL: vllm died. ROOT CAUSE (first real error in the log):"
    # The outer traceback always ends in "Engine core initialization failed.
    # See root cause above." -- so surface what is actually above it.
    # Worker errors are prefixed "(Worker pid=NNN) ERROR ... [file:line] " so a
    # ^-anchored pattern never matches. Match the exception ANYWHERE in the line.
    grep -oE "[A-Za-z_][A-Za-z_.]*(Error|Exception|Interrupt): ?[^\"]{0,160}" \
      "$ROOT/logs/${RUN_NAME}_vllm.log" 2>/dev/null \
      | grep -viE "^RuntimeError: Engine core initialization" | sort -u | tail -10 | sed "s/^/    !! /"
    grep -oE "No module named [^ ]+|CUDA out of memory|invalid device ordinal|no kernel image|NCCL error|free memory[^,]*" "$ROOT/logs/${RUN_NAME}_vllm.log" 2>/dev/null | sort -u | head -6 | sed "s/^/    >> /"
    echo "  --- last 25 lines ---"
    tail -25 "$ROOT/logs/${RUN_NAME}_vllm.log" | sed "s/^/    /"
    echo "  full log: $ROOT/logs/${RUN_NAME}_vllm.log"
    exit 1; }
  kill -0 "$SEARCH_PID" 2>/dev/null || { echo "FATAL: search died"; tail -40 "$ROOT/logs/${RUN_NAME}_search.log"; exit 1; }
  sleep 10
done
curl -s -m 3 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1 \
  || { echo "FATAL: vllm never became ready"; tail -40 "$ROOT/logs/${RUN_NAME}_vllm.log"; exit 1; }
grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" \
  "$ROOT/logs/${RUN_NAME}_vllm.log" | tail -1 | sed 's/^/[kv] /'

# ---------------------------------------------------------------- agent
cd "$ROOT/scaffold/OpenResearcher"
python -u deploy_agent.py \
  --output_dir "$OUT" \
  --model_name_or_path "$MODEL" \
  --search_url "http://127.0.0.1:$SEARCH_PORT" \
  --dataset_name browsecomp_plus \
  --data_path "$ROOT/data/browsecomp-plus/data/*.parquet" \
  --browser_backend local \
  --vllm_server_url "http://127.0.0.1:$VLLM_PORT/v1" \
  --max_concurrency_per_worker "$CONC" \
  "$@" 2>&1 | tee -a "$ROOT/logs/${RUN_NAME}_agent.log" &
AGENT_PID=$!
wait $AGENT_PID
cd "$ROOT"

N=$(cat "$OUT"/node_*_shard_*.jsonl 2>/dev/null | wc -l)
F=$(grep -ho '"status": "fail"' "$OUT"/node_*_shard_*.jsonl 2>/dev/null | wc -l)
echo
echo "=========================================================="
echo "finished: $N/830 done, $F failed"
echo "  re-run the same command to continue if < 830"
echo "  then check quality:  bash 4_check.sh $CKPT"
echo "=========================================================="
