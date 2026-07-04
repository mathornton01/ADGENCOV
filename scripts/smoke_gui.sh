#!/usr/bin/env bash
# Phase D end-to-end smoke test: boot the FastAPI service, verify it serves the
# static SPA (index.html + app.js + styles.css), drive a real upload analysis
# through the HTTP job API, confirm the covariance payload the heatmap needs is
# present, then tear everything down. One atomic run — no handoff.
set -Eeuo pipefail

ROOT="/home/micah/herald-workspace/herald/adgencov"
VENV="$ROOT/.goldenv"
export PYTHONPATH="$ROOT/python"
HOST="127.0.0.1"; PORT="8137"; BASE="http://$HOST:$PORT"
LOG="$(mktemp)"; SRV_PID=""

cleanup(){ [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

step(){ printf '\n\033[1;36m== %s\033[0m\n' "$1"; }
ok(){   printf '   \033[1;32mOK\033[0m  %s\n' "$1"; }
die(){  printf '   \033[1;31mFAIL\033[0m %s\n' "$1"; echo "--- server log ---"; cat "$LOG"; exit 1; }

# --- locate a real expression fixture to drive the analysis ------------------
step "Locating expression fixture"
FIX="$(find "$ROOT" -name '*expr*fixture*.tsv' -o -name 'expr_fixture.tsv' 2>/dev/null | head -1 || true)"
[ -z "$FIX" ] && FIX="$(find "$ROOT" \( -name '*.tsv' -o -name '*.csv' \) -path '*fixture*' 2>/dev/null | head -1 || true)"
[ -z "$FIX" ] && FIX="$(find "$ROOT" \( -name '*.tsv' -o -name '*.csv' \) 2>/dev/null | grep -iE 'expr|matrix|fixture|test' | head -1 || true)"
[ -z "$FIX" ] && die "no expression fixture found under $ROOT"
ok "fixture: $FIX"
GENE_COL="$(head -1 "$FIX" | tr '\t,' '\n\n' | head -1)"
ok "gene column: $GENE_COL"

# --- launch the service (detached, correct venv + PYTHONPATH) ----------------
step "Launching service on $BASE"
"$VENV/bin/python" -m uvicorn "adgencov.api.app:create_app" --factory \
  --host "$HOST" --port "$PORT" --log-level warning >"$LOG" 2>&1 &
SRV_PID=$!
ok "pid $SRV_PID"

step "Waiting for boot"
for i in $(seq 1 50); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then ok "up after ${i}x0.3s"; break; fi
  kill -0 "$SRV_PID" 2>/dev/null || die "server died during boot"
  sleep 0.3
  [ "$i" = 50 ] && die "server did not come up"
done

# --- health -----------------------------------------------------------------
step "Health"
H="$(curl -fsS "$BASE/health")"; echo "   $H"
echo "$H" | grep -q '"status":"ok"' || die "health not ok"
ok "healthy"

# --- static SPA served ------------------------------------------------------
step "Static dashboard (SPA) served"
IDX="$(curl -fsS "$BASE/")"
echo "$IDX" | grep -qi 'ADGENCOV' || die "index.html did not serve"
echo "$IDX" | grep -q 'app.js'    || die "index.html missing app.js reference"
ok "GET / -> index.html ($(printf '%s' "$IDX" | wc -c) bytes)"
curl -fsS "$BASE/app.js"     | grep -q 'ADGENCOV'  || die "app.js did not serve"
ok "GET /app.js served"
curl -fsS "$BASE/styles.css" | grep -q 'topbar'    || die "styles.css did not serve"
ok "GET /styles.css served"

# --- drive a real analysis through the async job API ------------------------
step "Submitting upload analysis"
SUB="$(curl -fsS -X POST "$BASE/analyze/upload" \
  -F "file=@$FIX" -F "n_genes=200" -F "group=correlation_blocks" \
  -F "n_blocks=3" -F "top_fraction=0.05" -F "gene_col=$GENE_COL")"
echo "   $SUB"
JOB="$(echo "$SUB" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
[ -z "$JOB" ] && die "no job id returned"
ok "job $JOB accepted (202)"

step "Polling job to completion"
STATE=""; RESULT=""
for i in $(seq 1 100); do
  DETAIL="$(curl -fsS "$BASE/jobs/$JOB")"
  STATE="$(echo "$DETAIL" | sed -n 's/.*"state":"\([^"]*\)".*/\1/p')"
  case "$STATE" in
    succeeded) RESULT="$DETAIL"; ok "succeeded after ${i} polls"; break;;
    failed)    die "job failed: $(echo "$DETAIL" | sed -n 's/.*"error":"\([^"]*\)".*/\1/p')";;
  esac
  sleep 0.3
  [ "$i" = 100 ] && die "job did not finish (last state: $STATE)"
done

# --- validate the payload the GUI renders -----------------------------------
step "Validating analysis payload (what the dashboard renders)"
"$VENV/bin/python" - "$RESULT" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
r = d["result"]
assert r["recommended"], "no recommended estimator"
assert r["ranking"] and "loo_nll" in r["ranking"][0], "ranking malformed"
assert isinstance(r["genes"], list) and r["genes"], "no genes"
assert isinstance(r["labels"], list) and len(r["labels"]) == len(r["genes"]), "labels/genes mismatch"
cov = r["covariance"]
assert cov and len(cov) == len(cov[0]) == r["n_genes"], "covariance not square p x p"
print(f"   recommended : {r['recommended']}")
print(f"   genes       : {r['n_genes']}")
print(f"   blocks      : {len(set(r['labels']))}")
print(f"   covariance  : {len(cov)}x{len(cov[0])} (heatmap-ready)")
print(f"   edges       : {len(r['edges'])}")
print(f"   top method  : {r['ranking'][0]['method']}  LOO-NLL={r['ranking'][0]['loo_nll']:.4f}")
PY
[ $? -eq 0 ] && ok "payload valid: recommendation, ranking, blocks, square covariance, edges"

# --- job list + delete (the sidebar's operations) ---------------------------
step "Job list & delete"
curl -fsS "$BASE/jobs" | grep -q "$JOB" || die "job not in list"
ok "job present in GET /jobs"
curl -fsS -o /dev/null -w '%{http_code}' -X DELETE "$BASE/jobs/$JOB" | grep -q 204 || die "delete failed"
ok "DELETE /jobs/$JOB -> 204"

printf '\n\033[1;32m========== PHASE D GUI SMOKE TEST: ALL GREEN ==========\033[0m\n'
