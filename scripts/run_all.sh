#!/usr/bin/env bash
# One experiment run = one folder: runs/<member>/<YYYYmmdd-HHMM>_<tag>/
#
#   MEMBER=yschoi TAG=baseline scripts/run_all.sh                       # dev split
#   MEMBER=yschoi TAG=final SPLIT=test CONFIRM_TEST=1 scripts/run_all.sh  # held-out test, once
#   MEMBER=jmkim  TAG=hang2 NOTE="speech dilation 2 frames" scripts/run_all.sh --hang 2
#   MEMBER=sgkwon TAG=quick N_DIALOGS=20 SKIP_VERIFY=1 scripts/run_all.sh --budgets 125 250
#   RUN_DIR=runs/yschoi/20261110-0510_baseline scripts/run_all.sh     # resume an interrupted run
#
# Arguments after the script name go to scripts/03_run_nll.py (policy grid options).
# Env vars:
#   MEMBER      required: an id from members.json (sgkwon | jmkim | yschoi)
#   TAG         short run name (default "run"); NOTE free-text memo saved in meta.json
#   GPU_PAIRS   workers, "devA,devB" each (model split over 2 GPUs; "0,0" for one 24GB GPU).
#               default: the member's gpu_pairs in members.json, else auto-detect
#   EXP         kv_policy (default) | rvq | kv_quant   (experiment type; passed to 03_run_nll.py --exp)
#   NEW_RUN_DIR explicit run folder (the experiment console sets this)
#   SPLIT       dev (default) | test. dev = tuning/exploration. test = frozen final evaluation:
#               needs CONFIRM_TEST=1 and is logged in splits/test_log.jsonl (run it once per final method)
#   N_DIALOGS   first N dialogs of the split (default: all 100)   RATIOS silence ratios (default "0.2 0.4 0.6")
#   SKIP_VERIFY 1 = skip scripts/02_verify.py
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$PWD/hf_cache}
PY=.venv/bin/python
SPLIT=${SPLIT:-dev}
EXP=${EXP:-kv_policy}
N_DIALOGS=${N_DIALOGS:-100}
[ -f "splits/$SPLIT.txt" ] || { echo "unknown SPLIT=$SPLIT (splits/dev.txt, splits/test.txt)"; exit 1; }
RATIOS=${RATIOS:-"0.2 0.4 0.6"}
NLL_ARGS=("$@")

if [ -n "${RUN_DIR:-}" ]; then
  [ -f "$RUN_DIR/meta.json" ] || { echo "RUN_DIR=$RUN_DIR has no meta.json"; exit 1; }
  # resume with the run's own settings, not the current env
  eval "$("$PY" - "$RUN_DIR" <<'EOF'
import json, shlex, sys
m = json.load(open(sys.argv[1] + "/meta.json"))
print(f"MEMBER={shlex.quote(m['member'])}; SPLIT={shlex.quote(m.get('split', 'dev'))}; EXP={shlex.quote(m.get('exp', 'kv_policy'))}; "
      f"N_DIALOGS={m['n_dialogs']}; RATIOS={shlex.quote(' '.join(str(r) for r in m['ratios']))}; "
      f"GPU_PAIRS=${{GPU_PAIRS:-{shlex.quote(m['gpu_pairs'])}}}; "
      f"NLL_ARGS=({' '.join(shlex.quote(a) for a in m.get('nll_args', []))})")
EOF
)"
  echo "resuming $RUN_DIR (split $SPLIT)"
else
  : "${MEMBER:?set MEMBER to one of: $("$PY" -c "import json;print(' '.join(json.load(open('members.json'))['members']))")}"
  "$PY" -c "import json,sys;m=json.load(open('members.json'))['members'];sys.exit(0 if '$MEMBER' in m else 'unknown MEMBER $MEMBER, see members.json')"
  TAG=${TAG:-run}
  if [ "$SPLIT" = test ]; then
    if [ "${CONFIRM_TEST:-0}" != 1 ]; then
      echo "SPLIT=test is the held-out set: evaluate it only with final, frozen settings."
      echo "previous test evaluations:"; cat splits/test_log.jsonl 2>/dev/null || echo "  (none)"
      echo "re-run with CONFIRM_TEST=1 to proceed."; exit 1
    fi
  fi
  RUN_DIR=${NEW_RUN_DIR:-runs/$MEMBER/$(date +%Y%m%d-%H%M)_${SPLIT}_${TAG}}
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/nll"

if [ -z "${GPU_PAIRS:-}" ]; then
  GPU_PAIRS=$("$PY" -c "import json;print(json.load(open('members.json'))['members']['$MEMBER'].get('gpu_pairs',''))")
fi
if [ -z "$GPU_PAIRS" ]; then
  GPU_PAIRS=$("$PY" - <<'EOF'
import torch
n = torch.cuda.device_count()
assert n, "no CUDA GPU found"
mem = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(n)]
if min(mem) >= 22:
    print(" ".join(f"{i},{i}" for i in range(n)))
else:
    assert n >= 2, "a single GPU under 22GB cannot hold moshiko fp16 (~15.4GB + activations)"
    print(" ".join(f"{i},{i+1}" for i in range(0, n - 1, 2)))
EOF
)
fi
read -ra PAIRS <<< "$GPU_PAIRS"
NW=${#PAIRS[@]}
FIRST_A=${PAIRS[0]%,*}; FIRST_B=${PAIRS[0]#*,}

# meta.json: who / what / which code. Written before running so failed runs are traceable too.
if [ ! -f "$RUN_DIR/meta.json" ]; then
  EXP="$EXP" MEMBER="$MEMBER" TAG="${TAG:-run}" SPLIT="$SPLIT" NOTE="${NOTE:-}" GPU_PAIRS="$GPU_PAIRS" N_DIALOGS="$N_DIALOGS" RATIOS="$RATIOS" \
  "$PY" - "$RUN_DIR" "${NLL_ARGS[@]}" <<'EOF'
import json, os, socket, subprocess, sys, datetime
run_dir, nll_args = sys.argv[1], sys.argv[2:]
def git(*a):
    try: return subprocess.check_output(["git", *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
members = json.load(open("members.json"))["members"]
meta = {
    "member": os.environ["MEMBER"], "name": members[os.environ["MEMBER"]]["name"],
    "tag": os.environ["TAG"], "note": os.environ["NOTE"], "split": os.environ["SPLIT"], "exp": os.environ["EXP"],
    "started": datetime.datetime.now().isoformat(timespec="seconds"), "finished": None, "status": "running",
    "host": socket.gethostname(), "gpu_pairs": os.environ["GPU_PAIRS"],
    "n_dialogs": int(os.environ["N_DIALOGS"]), "ratios": [float(r) for r in os.environ["RATIOS"].split()],
    "nll_args": nll_args,
    "git_commit": git("rev-parse", "--short", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "fdkv", "scripts")),
}
json.dump(meta, open(f"{run_dir}/meta.json", "w"), ensure_ascii=False, indent=2)
if meta["split"] == "test":
    with open("splits/test_log.jsonl", "a") as f:
        f.write(json.dumps({k: meta[k] for k in ("member", "tag", "started", "git_commit", "git_dirty", "nll_args", "note")}
                           | {"run": run_dir}, ensure_ascii=False) + "\n")
EOF
fi
set_status() { "$PY" -c "
import json,datetime;p='$RUN_DIR/meta.json';m=json.load(open(p));m['status']='$1'
m['finished']=datetime.datetime.now().isoformat(timespec='seconds') if '$1'!='running' else None
json.dump(m,open(p,'w'),ensure_ascii=False,indent=2)"; }
trap 'set_status failed' ERR

echo "run: $RUN_DIR  member: $MEMBER  split: $SPLIT  workers: $NW ($GPU_PAIRS)"
# first run on a machine: fetch model + this split's dialogs if missing (data is not in git)
if ! "$PY" - "$SPLIT" <<'EOF'
import glob, json, os, sys
os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
ids = open(f"splits/{sys.argv[1]}.txt").read().split()
root = "data/DailyTalkContiguous"
have_data = all(os.path.exists(f"{root}/data_stereo/{i}.wav") and os.path.exists(f"{root}/data_stereo/{i}.json") for i in ids) \
    and os.path.exists(f"{root}/dailytalk.jsonl")
have_model = bool(glob.glob(os.path.join(os.environ["HF_HOME"], "hub/models--kyutai--moshiko-pytorch-bf16/snapshots/*/model.safetensors")))
sys.exit(0 if have_data and have_model else 1)
EOF
then
  echo "[0/4] model or dialogs missing -> downloading (scripts/00_download.py)"
  $PY scripts/00_download.py
fi
echo "[1/4] data prep (shared data/prepared) -> $RUN_DIR/ids.txt"
$PY scripts/01_prep_data.py --ids_file "splits/$SPLIT.txt" --n "$N_DIALOGS" --ratios $RATIOS --gpu "$FIRST_A" \
    --ids_out "$RUN_DIR/ids.txt" > "$RUN_DIR/logs/prep.log" 2>&1

if [ "${SKIP_VERIFY:-0}" != "1" ]; then
  echo "[2/4] correctness check"
  R_LAST=$(echo $RATIOS | awk '{print $NF}')
  $PY scripts/02_verify.py --data "data/prepared/r${R_LAST}" --gpus "$FIRST_A" "$FIRST_B" > "$RUN_DIR/logs/verify.log" 2>&1
  grep -E "^\(|  " "$RUN_DIR/logs/verify.log" | grep -v -i warn || true
fi

echo "[3/4] NLL grid (logs: $RUN_DIR/logs/nll_shard*.log)"
for i in "${!PAIRS[@]}"; do
  A=${PAIRS[$i]%,*}; B=${PAIRS[$i]#*,}
  $PY scripts/03_run_nll.py --data data/prepared --out "$RUN_DIR/nll" --gpus "$A" "$B" \
      --shard "$i" --nshards "$NW" --batch 3 --exp "$EXP" --ids_file "$RUN_DIR/ids.txt" --ratios $RATIOS \
      "${NLL_ARGS[@]}" > "$RUN_DIR/logs/nll_shard$i.log" 2>&1 &
done
FAIL=0; for job in $(jobs -p); do wait "$job" || FAIL=1; done
[ "$FAIL" = 0 ] || { echo "a worker failed, see $RUN_DIR/logs/"; set_status failed; exit 1; }

$PY -m fdkv.results "$RUN_DIR/nll" > /dev/null   # shards -> nll/summary.jsonl.gz (the only result file in git)
echo "[4/4] analysis -> $RUN_DIR/nll/analysis/"
$PY scripts/07_analyze_variants.py --res "$RUN_DIR/nll" > "$RUN_DIR/nll/analysis.txt"
if [ "$EXP" = kv_policy ]; then
  $PY scripts/04_analyze.py --res "$RUN_DIR/nll" >> "$RUN_DIR/nll/analysis.txt"
  $PY scripts/06_hypotheses.py --res "$RUN_DIR/nll" | tail -25
else
  tail -25 "$RUN_DIR/nll/analysis.txt"
fi
set_status done
echo "DONE: $RUN_DIR   (compare: .venv/bin/python scripts/runs.py)"
