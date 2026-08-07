#!/bin/bash
# Pick usable GPUs on a SHARED node, at launch time.
#
# Prints a CUDA_VISIBLE_DEVICES list on stdout (e.g. "0,1,2,3"); everything else
# goes to stderr so callers can do:  DEVS=$(bash pick_gpus.sh)
#
# Rules:
#   - a GPU is "usable" only if it has >= MIN_FREE_GIB free (default 12)
#   - we take at most MAX_GPUS (default 4) so we never hog a shared box
#   - tensor-parallel requires a count that divides the model's 8 KV heads,
#     so the final count is rounded DOWN to 1, 2, or 4
#   - prefers the emptiest GPUs, so we sit beside other jobs rather than on them
#
# Env overrides:  MIN_FREE_GIB, MAX_GPUS
set -uo pipefail
MIN_FREE_GIB="${MIN_FREE_GIB:-12}"
MAX_GPUS="${MAX_GPUS:-4}"

command -v nvidia-smi >/dev/null || { echo "0"; echo "[gpu] nvidia-smi not found, defaulting to GPU 0" >&2; exit 0; }

# index,total,used  -> compute free
MAP=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', *' '{printf "%s %s\n", $1, ($2-$3)/1024}')

echo "[gpu] free VRAM per device (need >= ${MIN_FREE_GIB} GiB to be usable):" >&2
echo "$MAP" | while read -r i f; do
  st="unusable"
  awk -v f="$f" -v m="$MIN_FREE_GIB" 'BEGIN{exit !(f>=m)}' && st="USABLE"
  printf "[gpu]   GPU %-2s %6.1f GiB  %s\n" "$i" "$f" "$st" >&2
done

# usable, sorted by most-free first
USABLE=$(echo "$MAP" | awk -v m="$MIN_FREE_GIB" '$2>=m {print $2" "$1}' | sort -rn | awk '{print $2}')
N=$(echo "$USABLE" | grep -c . || true)

if [ "${N:-0}" -eq 0 ]; then
  echo "[gpu] FATAL: no GPU has >= ${MIN_FREE_GIB} GiB free." >&2
  echo "[gpu]   DR-Tulu needs ~15.3 GiB of weights; with TP=4 that is ~3.8 GiB/GPU" >&2
  echo "[gpu]   plus KV cache. Wait for capacity, or lower MIN_FREE_GIB if you" >&2
  echo "[gpu]   know the other jobs will not grow." >&2
  exit 1
fi

# cap, then round down to a valid tensor-parallel size (8 KV heads -> 1,2,4,8)
[ "$N" -gt "$MAX_GPUS" ] && N="$MAX_GPUS"
if   [ "$N" -ge 4 ]; then TP=4
elif [ "$N" -ge 2 ]; then TP=2
else TP=1
fi

DEVS=$(echo "$USABLE" | head -n "$TP" | sort -n | paste -sd, -)
echo "[gpu] using $TP GPU(s): $DEVS   (cap ${MAX_GPUS}, leaving the rest for other users)" >&2
echo "$DEVS"
