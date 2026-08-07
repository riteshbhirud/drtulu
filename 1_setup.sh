#!/bin/bash
# DR-Tulu BrowseComp-Plus bundle — STEP 1: build the environment.
# Run once on the target machine. Safe to re-run (skips finished steps).
#
#   bash 1_setup.sh
#
# Needs: python >= 3.12 (auto-installed via uv if absent), java 21 (auto), ~120 GB disk, internet.
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
# We need >= 3.12: the scaffold's browser tool imports gpt_oss, and EVERY
# gpt-oss release on PyPI requires-python >= 3.12. 3.10 cannot install it.
PYBIN=""
for c in python3.13 python3.12 python3; do
  command -v "$c" >/dev/null || continue
  if "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,12) else 1)' 2>/dev/null; then
    PYBIN="$(command -v "$c")"; break
  fi
done
if [ -z "$PYBIN" ] && [ -x "$ROOT/py312/bin/python3" ]; then
  PYBIN="$ROOT/py312/bin/python3"
fi
if [ -z "$PYBIN" ]; then
  echo "   system python is $(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])'), need >= 3.12"
  echo "   installing a local Python 3.12 (no root) via uv ..."
  if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
  fi
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  if [ -x "$UV" ]; then
    "$UV" python install 3.12 2>&1 | tail -1
    # `uv python find` can return a python already on PATH rather than the one
    # it just installed, so look in uv's own install dir first.
    PYBIN="$(ls -1 "$HOME"/.local/share/uv/python/cpython-3.1[2-9]*/bin/python3 2>/dev/null | head -1)"
    [ -z "$PYBIN" ] && PYBIN="$("$UV" python find 3.12 2>/dev/null || true)"
    # verify whatever we found is really >= 3.12
    if [ -n "$PYBIN" ] && ! "$PYBIN" -c 'import sys;sys.exit(0 if sys.version_info>=(3,12) else 1)' 2>/dev/null; then
      PYBIN=""
    fi
  fi
  if [ -z "$PYBIN" ] || [ ! -x "$PYBIN" ]; then
    echo "FATAL: could not obtain Python >= 3.12."
    echo "   The scaffold needs gpt-oss, which has no build for python < 3.12."
    echo "   Options:"
    echo "     conda create -y -p $ROOT/env python=3.12 && bash 1_setup.sh"
    echo "     module load python/3.12   (if your cluster has modules)"
    exit 1
  fi
  echo "   using local python: $PYBIN"
fi
PYV=$("$PYBIN" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "   python $PYV  ($PYBIN)"

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
# If an env already exists but was built on the wrong python (e.g. a 3.10 run
# before we discovered gpt-oss needs >=3.12), rebuild it rather than silently
# reusing a version that cannot install the dependencies.
if [ -x "$ROOT/env/bin/python" ]; then
  if ! "$ROOT/env/bin/python" -c 'import sys;sys.exit(0 if sys.version_info>=(3,12) else 1)' 2>/dev/null; then
    echo "   existing env is python $("$ROOT/env/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])') -- rebuilding on $PYV"
    rm -rf "$ROOT/env"
  fi
fi
if [ ! -x "$ROOT/env/bin/python" ]; then
  MADE=0
  # (a) stock venv -- works when ensurepip is present
  if [ $MADE -eq 0 ] && "$PYBIN" -m venv "$ROOT/env" 2>/dev/null && [ -x "$ROOT/env/bin/python" ]; then
    echo "   created with: python3 -m venv"; MADE=1
  fi
  # (b) venv without pip, then bootstrap pip via get-pip.py
  if [ $MADE -eq 0 ]; then
    rm -rf "$ROOT/env"
    if "$PYBIN" -m venv --without-pip "$ROOT/env" 2>/dev/null && [ -x "$ROOT/env/bin/python" ]; then
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
    "$PYBIN" -m pip install --user -q virtualenv 2>/dev/null || true
    if "$PYBIN" -m virtualenv "$ROOT/env" 2>/dev/null && [ -x "$ROOT/env/bin/python" ]; then
      echo "   created with: virtualenv"; MADE=1
    fi
  fi
  # (d) conda / micromamba
  if [ $MADE -eq 0 ] && command -v conda >/dev/null; then
    rm -rf "$ROOT/env"
    conda create -y -p "$ROOT/env" python=3.12 pip >/dev/null 2>&1 \
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
# vllm 0.26 needs setuptools <81 and torch 2.11 needs <82; an unpinned
# --upgrade pulls 83 and breaks both.
python -m pip install -q --upgrade pip wheel
python -m pip install -q "setuptools>=77.0.3,<81.0.0"

# ---------------------------------------------------------------- deps
echo "[3/6] python packages"
echo "   NOTE: vllm pulls torch + several GB of CUDA libraries."
echo "         Expect 10-30 min. Progress is shown below; if it looks stalled,"
echo "         check from another shell:  du -sh $ROOT/env"
if ! python -c "import vllm" 2>/dev/null; then
  # Deliberately NOT -q: a silent multi-GB download is indistinguishable from
  # a hang, and that ambiguity has already cost us time.
  pip install --progress-bar off vllm 2>&1 | grep -E "Collecting|Downloading|Installing|Successfully|ERROR" | tail -40
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
# hf_xet is a Rust transfer backend that segfaults on some network
# filesystems. Remove it so nothing can load it, env var or not.
python -m pip uninstall -y -q hf_xet hf_transfer 2>/dev/null || true

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
# huggingface-cli was REMOVED in recent huggingface_hub; the command is now `hf`.
# Try the modern entrypoint first and fall back through the older spellings.
# Do NOT swallow stderr: a failed download used to look like success.
# hf_xet is a Rust transfer backend that SEGFAULTS on some network
# filesystems (seen on /netdisk). Disable it and the older hf_transfer so
# downloads use plain HTTP, which is slower but does not crash.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

hf() {
  if command -v hf >/dev/null 2>&1; then
    hf "$@"
  elif python -c "import huggingface_hub.commands" 2>/dev/null; then
    python -m huggingface_hub.commands.huggingface_cli "$@"
  else
    huggingface-cli "$@"
  fi
}

# Pure-python download, used when the CLI crashes (segfault, etc).
py_download() {  # py_download <repo> <local_dir> [allow_pattern]
  python - "$1" "$2" "${3:-}" <<'PYEOF'
import os, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
repo, local, pat = sys.argv[1], sys.argv[2], sys.argv[3]
kw = dict(repo_id=repo, repo_type="dataset", local_dir=local,
          max_workers=4, tqdm_class=None)
if pat:
    kw["allow_patterns"] = [pat]
snapshot_download(**kw)
print("   downloaded", repo)
PYEOF
}

# Try the CLI; if it dies for ANY reason (including a segfault, which shows up
# as exit 139), fall back to the python API.
hf_get() {  # hf_get <repo> <local_dir> [allow_pattern]
  # Python API ONLY. The `hf` CLI segfaults on network filesystems (hf_xet
  # Rust backend); a crashing binary in the path just costs a round trip.
  py_download "$1" "$2" "${3:-}"
}

# Skip only when the expected FILES exist -- an empty directory left behind by a
# failed download previously caused this step to be skipped forever, and the run
# then died at service startup with "No files found that match the pattern".
need_dl() {  # need_dl <glob> ; true when the glob matches nothing
  ! compgen -G "$1" > /dev/null 2>&1
}

# Fail early and clearly if no huggingface CLI is usable at all.
if ! command -v hf >/dev/null 2>&1 \
   && ! python -c "import huggingface_hub.commands" 2>/dev/null \
   && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "FATAL: no usable huggingface CLI found."
  echo "   pip install -U huggingface_hub   then re-run 1_setup.sh"
  exit 1
fi

if need_dl "$ROOT/data/browsecomp-plus/data/*.parquet"; then
  echo "   downloading queries..."
  hf_get Tevatron/browsecomp-plus "$ROOT/data/browsecomp-plus" || true
fi
if need_dl "$ROOT/data/browsecomp-plus-corpus/data/*.parquet"; then
  echo "   downloading corpus (~1.7 GB)..."
  hf_get Tevatron/browsecomp-plus-corpus "$ROOT/data/browsecomp-plus-corpus" || true
fi
if need_dl "$ROOT/data/browsecomp-plus-indexes/bm25/*"; then
  echo "   downloading BM25 index (~2.1 GB)..."
  hf_get Tevatron/browsecomp-plus-indexes "$ROOT/data/browsecomp-plus-indexes" "bm25/*" || true
fi

# Hard gate: the run cannot work without these, so fail here rather than at
# service startup 10 minutes later.
DATA_OK=1
for g in "$ROOT/data/browsecomp-plus/data/*.parquet" \
         "$ROOT/data/browsecomp-plus-corpus/data/*.parquet" \
         "$ROOT/data/browsecomp-plus-indexes/bm25/*"; do
  if compgen -G "$g" > /dev/null 2>&1; then
    n=$(compgen -G "$g" | wc -l)
    echo "     OK   $n file(s): $(dirname "$g")"
  else
    echo "     MISSING: $g"
    DATA_OK=0
  fi
done
if [ "$DATA_OK" -eq 0 ]; then
  echo "FATAL: dataset download incomplete. Re-run 1_setup.sh (it resumes),"
  echo "       or check network/HF access. Do not run 3_run.sh until this passes."
  exit 1
fi
du -sh "$ROOT"/data/* 2>/dev/null | sed 's/^/     /'

# ---------------------------------------------------------------- models
echo "[6/6] DR-Tulu checkpoints (~48 GB total)"
mkdir -p "$ROOT/models"
dl() {
  local repo="$1" dir="$2"
  if [ -f "$ROOT/models/$dir/.complete" ]; then echo "     [skip] $dir"; return; fi
  echo "     downloading $repo (~16 GB, python API; progress below)"
  python - "$repo" "$ROOT/models/$dir" <<'PYEOF' && touch "$ROOT/models/$dir/.complete"
import os, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], max_workers=4)
print("     downloaded", sys.argv[1])
PYEOF
}
dl rl-research/DR-Tulu-8B      DR-Tulu-8B
dl rl-research/DR-Tulu-SFT-8B  DR-Tulu-SFT-8B
dl Qwen/Qwen3-8B               Qwen3-8B

MODELS_OK=1
for m in DR-Tulu-8B DR-Tulu-SFT-8B Qwen3-8B; do
  if [ -f "$ROOT/models/$m/config.json" ] && compgen -G "$ROOT/models/$m/*.safetensors" >/dev/null 2>&1; then
    echo "     OK   models/$m"
  else
    echo "     INCOMPLETE: models/$m"; MODELS_OK=0
  fi
done
[ "$MODELS_OK" -eq 1 ] || { echo "FATAL: model download incomplete. Re-run 1_setup.sh (it resumes)."; exit 1; }
du -sh "$ROOT"/models/* 2>/dev/null | sed 's/^/     /'

echo
echo "=========================================================="
echo "setup complete. NEXT:  bash 2_verify.sh"
echo "=========================================================="
