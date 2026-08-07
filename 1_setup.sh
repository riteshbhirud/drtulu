#!/bin/bash
# DR-Tulu BrowseComp-Plus bundle — STEP 1: build the environment.
# Run once on the target machine. Safe to re-run (skips finished steps).
#
#   bash 1_setup.sh
#
# Needs: python3.10+, java 21 (for pyserini/Lucene), ~120 GB disk, internet.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
echo "=========================================================="
echo "DR-Tulu bundle setup"
echo "root: $ROOT"
echo "=========================================================="

# ---------------------------------------------------------------- checks
echo "[1/6] preflight"
command -v python3 >/dev/null || { echo "FATAL: python3 not found"; exit 1; }
PYV=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "   python $PYV"
python3 -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' \
  || { echo "FATAL: need python >= 3.10"; exit 1; }

# pyserini/Lucene REQUIRE Java 21; Java 11 cannot open the index. No root needed
# -- we drop a JDK into the bundle and point JAVA_HOME at it.
NEED_JDK=1
if command -v java >/dev/null && java -version 2>&1 | grep -qE '"(21|22|23)'; then
  NEED_JDK=0
  echo "   java: $(java -version 2>&1 | head -1) [OK]"
elif [ -x "$ROOT/jdk21/bin/java" ]; then
  NEED_JDK=0
  echo "   java: bundled JDK 21 already present"
else
  echo "   java: $( (java -version 2>&1 | head -1) || echo 'not found') -- need 21, will install locally"
fi
if [ "$NEED_JDK" -eq 1 ]; then
  echo "   downloading Temurin JDK 21 (no root required)..."
  ARCH=$(uname -m); case "$ARCH" in x86_64) A=x64;; aarch64) A=aarch64;; *) A=x64;; esac
  URL="https://api.adoptium.net/v3/binary/latest/21/ga/linux/${A}/jdk/hotspot/normal/eclipse"
  if curl -sSL "$URL" -o /tmp/jdk21.tar.gz || wget -qO /tmp/jdk21.tar.gz "$URL"; then
    mkdir -p "$ROOT/jdk21" && tar xzf /tmp/jdk21.tar.gz -C "$ROOT/jdk21" --strip-components=1 \
      && echo "   installed JDK 21 -> $ROOT/jdk21" \
      || echo "   WARNING: JDK extract failed; BM25 will not work"
  else
    echo "   WARNING: JDK download failed. Install Java 21 manually, e.g."
    echo "            conda install -c conda-forge openjdk=21"
  fi
fi
# Every later step sources this so JAVA_HOME is consistent.
if [ -x "$ROOT/jdk21/bin/java" ]; then
  cat > "$ROOT/env.sh" <<EOF
export JAVA_HOME="$ROOT/jdk21"
export PATH="\$JAVA_HOME/bin:\$PATH"
EOF
else
  cat > "$ROOT/env.sh" <<EOF
export JAVA_HOME="\${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v java 2>/dev/null)" 2>/dev/null)")" 2>/dev/null)}"
EOF
fi
# shellcheck disable=SC1091
source "$ROOT/env.sh" 2>/dev/null || true
echo "   JAVA_HOME=${JAVA_HOME:-<unset>}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null \
  | sed 's/^/   GPU /' || echo "   WARNING: nvidia-smi not available"

FREE_GB=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "   free disk: ${FREE_GB:-?} GB (need ~120 for 3 checkpoints + index)"

# ---------------------------------------------------------------- venv
echo "[2/6] python environment"
# A PIP_USER=1 in the environment makes get-pip.py (and every later pip call)
# fail with "Can not perform a '--user' install" inside a venv. Clear the whole
# family up front -- this cost us a failed setup once already.
unset PIP_USER PIP_TARGET PIP_PREFIX PYTHONHOME 2>/dev/null || true
export PIP_USER=0
# Many clusters ship python3 without ensurepip (Debian splits it into
# python3-venv), so `python3 -m venv` fails and you cannot apt-install without
# root. Try four paths in order and use whichever works.
if [ ! -x "$ROOT/env/bin/python" ]; then
  MADE=0
  # (a) stock venv -- works when ensurepip is present
  if [ $MADE -eq 0 ] && python3 -m venv "$ROOT/env" 2>/dev/null && [ -x "$ROOT/env/bin/python" ]; then
    echo "   created with: python3 -m venv"; MADE=1
  fi
  # (b) venv without pip, then bootstrap pip via get-pip.py
  if [ $MADE -eq 0 ]; then
    rm -rf "$ROOT/env"
    if python3 -m venv --without-pip "$ROOT/env" 2>/dev/null && [ -x "$ROOT/env/bin/python" ]; then
      echo "   created with: python3 -m venv --without-pip"
      echo "   bootstrapping pip via get-pip.py ..."
      if curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$ROOT/get-pip.py" 2>/dev/null \
         || wget -qO "$ROOT/get-pip.py" https://bootstrap.pypa.io/get-pip.py 2>/dev/null; then
        # NOT -q: if this fails we need to see why, and a silent failure here
        # used to fall through and delete the env we had just built.
        if "$ROOT/env/bin/python" "$ROOT/get-pip.py" 2>&1 | tail -3; then :; fi
        if "$ROOT/env/bin/python" -m pip --version >/dev/null 2>&1; then
          echo "   pip bootstrapped: $("$ROOT/env/bin/python" -m pip --version)"
          MADE=1
        else
          echo "   get-pip.py ran but pip is still unusable"
        fi
      else
        echo "   could not download get-pip.py (no internet?)"
      fi
    fi
  fi
  # (c) the virtualenv package (bundles its own pip, no ensurepip needed)
  if [ $MADE -eq 0 ]; then
    rm -rf "$ROOT/env"
    python3 -m pip install --user -q virtualenv 2>/dev/null || true
    if python3 -m virtualenv "$ROOT/env" 2>/dev/null && [ -x "$ROOT/env/bin/python" ]; then
      echo "   created with: virtualenv"; MADE=1
    fi
  fi
  # (d) conda / micromamba
  if [ $MADE -eq 0 ] && command -v conda >/dev/null; then
    rm -rf "$ROOT/env"
    conda create -y -p "$ROOT/env" python=3.10 pip >/dev/null 2>&1 \
      && { echo "   created with: conda"; MADE=1; }
  fi
  if [ $MADE -eq 0 ]; then
    echo "FATAL: could not create a python environment. Options:"
    echo "   apt install python3.10-venv        (needs root)"
    echo "   python3 -m pip install --user virtualenv   then re-run"
    echo "   or load a conda module and re-run"
    exit 1
  fi
fi
# shellcheck disable=SC1091
source "$ROOT/env/bin/activate"
unset PIP_USER PIP_TARGET PIP_PREFIX 2>/dev/null || true
export PIP_USER=0
python -m pip --version >/dev/null 2>&1 || {
  echo "FATAL: pip is not usable inside $ROOT/env"
  echo "   If your shell exports PIP_USER=1, unset it and re-run:  unset PIP_USER"
  exit 1
}
echo "   using: $(python -V 2>&1), $(python -m pip --version | cut -d" " -f1-2)"
python -m pip install -q --upgrade pip setuptools wheel

# ---------------------------------------------------------------- deps
echo "[3/6] python packages (vllm pulls its own torch; this takes a while)"
if ! python -c "import vllm" 2>/dev/null; then
  pip install -q vllm
fi
# The scaffold imports these unconditionally, even on the pure-BM25 path.
for pkg in pyserini json5 duckdb python-dotenv prettytable faiss-cpu peft \
           "gpt-oss[all]>=0.0.9" openai-harmony aiohttp tqdm; do
  python - "$pkg" <<'PY' || pip install -q "$pkg"
import sys, importlib.util as u
name = sys.argv[1].split('[')[0].split('>=')[0].replace('-', '_')
alias = {"python_dotenv": "dotenv", "faiss_cpu": "faiss", "gpt_oss": "gpt_oss"}
sys.exit(0 if u.find_spec(alias.get(name, name)) else 1)
PY
done
# tevatron is imported by the scaffold's backend.py; install the vendored copy
python -c "import tevatron" 2>/dev/null || pip install -q --no-deps -e "$ROOT/scaffold/tevatron_src"

echo "   installed:"
python - <<'PY'
import importlib.metadata as md
for p in ["vllm","torch","pyserini","transformers","json5","faiss-cpu","peft"]:
    try: print(f"     {p:14s} {md.version(p)}")
    except Exception: print(f"     {p:14s} MISSING")
PY

# ---------------------------------------------------------------- lucene jars
echo "[4/6] lucene jars (pyserini highlighter)"
mkdir -p "$ROOT/scaffold/OpenResearcher/lucene_jars"
cd "$ROOT/scaffold/OpenResearcher/lucene_jars"
for j in lucene-highlighter lucene-queries lucene-memory; do
  [ -f "$j-9.9.1.jar" ] || wget -q "https://repo1.maven.org/maven2/org/apache/lucene/$j/9.9.1/$j-9.9.1.jar"
done
ls *.jar | sed 's/^/     /'
cd "$ROOT"

# ---------------------------------------------------------------- data
echo "[5/6] BrowseComp-Plus data (queries, corpus, BM25 index)"
mkdir -p "$ROOT/data"
python -m pip install -q huggingface_hub
hf() { python -m huggingface_hub.commands.huggingface_cli "$@" 2>/dev/null || huggingface-cli "$@"; }
[ -d "$ROOT/data/browsecomp-plus/data" ] || \
  hf download Tevatron/browsecomp-plus --repo-type dataset --local-dir "$ROOT/data/browsecomp-plus"
[ -d "$ROOT/data/browsecomp-plus-corpus/data" ] || \
  hf download Tevatron/browsecomp-plus-corpus --repo-type dataset --local-dir "$ROOT/data/browsecomp-plus-corpus"
[ -d "$ROOT/data/browsecomp-plus-indexes/bm25" ] || \
  hf download Tevatron/browsecomp-plus-indexes --repo-type dataset --include "bm25/*" \
      --local-dir "$ROOT/data/browsecomp-plus-indexes"
du -sh "$ROOT"/data/* 2>/dev/null | sed 's/^/     /'

# ---------------------------------------------------------------- models
echo "[6/6] DR-Tulu checkpoints (~48 GB total)"
mkdir -p "$ROOT/models"
dl() {
  local repo="$1" dir="$2"
  if [ -f "$ROOT/models/$dir/.complete" ]; then echo "     [skip] $dir"; return; fi
  echo "     downloading $repo"
  hf download "$repo" --local-dir "$ROOT/models/$dir" && touch "$ROOT/models/$dir/.complete"
}
dl rl-research/DR-Tulu-8B      DR-Tulu-8B
dl rl-research/DR-Tulu-SFT-8B  DR-Tulu-SFT-8B
dl Qwen/Qwen3-8B               Qwen3-8B
du -sh "$ROOT"/models/* 2>/dev/null | sed 's/^/     /'

echo
echo "=========================================================="
echo "setup complete. NEXT:  bash 2_verify.sh"
echo "=========================================================="
