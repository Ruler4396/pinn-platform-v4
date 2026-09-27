#!/usr/bin/env bash
# ops/run_second_impl.sh -- the SECOND-IMPLEMENTATION leg (登记七/登记九; the judgement is untouched).
#
# Venue rule, in the user's own words (登记九): NOTHING is installed on his machine. No solver, no
# conda/miniforge, no pip package, no WSL environment -- "zero cost" does not mean "free to install".
# The only local actions allowed are reading files, running stdlib self-tests and running gates that
# are already installed. The one work venue is the ModelScope instance, and anything installed goes
# to the NAS so it survives the box being swapped (/mnt/workspace is shared; everything under / is not).
#
# Paths and the per-trip output convention follow ops/run_k0b_5236655.sh ($WS is the instance checkout,
# $R2 the route2 script dir) rather than inventing a second layout in the same tree.
#
#   preflight  venue + blob hashes + what the output tree already holds -- fetches and installs nothing
#   install    DELEGATES to route2/install_external_solver.sh (one declaration of the conda command,
#              its own 60-min cap, its own tar+sha256+restore-line persistence)
#   smoke      solve_second_impl.py --nx 2 --ny 2: the REAL entry point on a 2x2 mesh, seconds.
#              Not a cross-check and never reported as one -- it proves the code path runs at all.
#   run        the base level (180x40, the .edp's own border counts) + crosscheck vs the shipped
#              FreeFEM truth. This is the row that decides the pre-registration's §4 verdict.
#   refine     the doubled level (360x80) + the same crosscheck + the internal <2% mesh gate
#   status     read-only: pointer, tree, env presence
#
# Every phase ends with one SEGMENT-END line: wall_used, the 8-hour window, what is missing, and
# whether the instance can be closed.  An idle box gets reaped, so work inside a segment rather than
# polling from outside; when the window is spent, hand the state back instead of silently continuing.

set -uo pipefail

NAS="${NAS:-/mnt/workspace}"                       # the shared mount; venue is proven against this
WS="${WS:-$NAS/pinn-repro-2026}"                   # the instance checkout (same name as run_k0b_5236655.sh)
R2="$WS/model/scripts/route2"
PREFIX="${PREFIX:-$WS/ext-fenics}"                 # passed explicitly to the installer, so the two
#                                                   # cannot land on different trees; at the default WS
#                                                   # it is install_external_solver.sh's own default
OUTROOT="${OUTROOT:-$WS/route2_out}"               # preflight prints what is already in here
REF_CSV="${REF_CSV:-$WS/model/cases/contraction_2d/cfd/C-base/C-base_raw.csv}"
MODE="${1:-status}"
SEGMENT_S="${SEGMENT_S:-1500}"                     # < 30 min per segment
WINDOW_S="${WINDOW_S:-28800}"                      # ~8 h ceiling, the user's number
T0=$(date +%s)
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "[ABORT] $*"; exit 3; }
seg_end() {  # seg_end <verdict> <what-is-missing>
  local wall=$(( $(date +%s) - T0 ))
  printf 'SEGMENT-END mode=%s verdict=%s wall_used=%ds segment_cap=%ss window_left=%ss prefix=%s out=%s missing=%s 实例可否关=%s\n' \
    "$MODE" "$1" "$wall" "$SEGMENT_S" "$(( WINDOW_S - wall ))" "$PREFIX" "${OUT:-none}" "$2" \
    "$([ "$1" = DONE ] && echo 可关 || echo 先别关-看missing)"
}

# --- venue: proven positively, once, and shared by every mode -------------------------------
# A bare [ -d "$NAS" ] is NOT a venue test. The portable-git root on the laptop provides its own
# /mnt, so the first rehearsal of this file passed that check on Windows and mkdir'd into there.
venue_check() {   # rc 0 = the instance; rc 1 = not, reason on stdout
  case "$(uname -s)" in
    Linux) : ;;
    *) printf 'uname=%s is not Linux' "$(uname -s)"; return 1 ;;
  esac
  if ! awk -v ws="$NAS" '$2 == ws || $2 == ws "/" {f = 1} END {exit !f}' /proc/mounts; then
    printf '%s is not a mount point here (mounts under /mnt: %s)' "$NAS" \
      "$(awk '$2 ~ /^\/mnt/ {printf "%s ", $2}' /proc/mounts 2>/dev/null)"
    return 1
  fi
}

require_venue() {
  local why
  why=$(venue_check) || die "$why -> not the instance, and the laptop is not a work venue (登记九: stop and report, never fall back here)"
  [ -d "$WS" ] || die "$WS absent -- the checkout is not there; report, do not re-fetch into a temp dir"
  touch "$NAS/.second_impl_wtest" 2>/dev/null || die "$NAS not writable"
  rm -f "$NAS/.second_impl_wtest"
}

PYBIN() { [ -x "$PREFIX/bin/python" ] && printf '%s' "$PREFIX/bin/python" || printf '%s' "$(command -v python3 || true)"; }

# The output directory is named ONCE by preflight and read back from a pointer by every other mode.
# "Take the newest timestamp" is how a trip reads the wrong tree; a missing pointer is an error, not
# a reason to guess.
read_pointer() {
  if [ -f "$OUTROOT/second_impl.POINTER" ]; then
    OUT="$(head -1 "$OUTROOT/second_impl.POINTER")"
    [ -d "$OUT" ] || die "pointer names a tree that is gone: $OUT"
    log "OUT from pointer = $OUT"
  fi
  # Without this, "${OUT}/smoke" silently becomes "/smoke" at the filesystem root -- a wrong path
  # that still writes, and still prints a plausible row count.
  [ -n "${OUT:-}" ] || die "no pointer at $OUTROOT/second_impl.POINTER and OUT not given -- run preflight first"
}

# --- what must be on disk before any of this runs ----------------------------------------------
# content_sha256 over the BLOB bytes (git show HEAD:<path>), which are LF. The instance checkout is a
# Linux clone with autocrlf off, so its working bytes equal the blob bytes there; on a CRLF working
# copy the two rulers part ways, which is why the number below is quoted with its ruler.
declare -A EXPECT=(
  [model/scripts/route2/solve_second_impl.py]=d8d05d7da4d60e9d
)
# crosscheck_second_impl.py and install_external_solver.sh are checked for PRESENCE only: they landed
# before this table existed, and their blobs are already in git (pin 7e67943 and earlier).
check_blobs() {
  local f miss="" got
  [ -f "$REF_CSV" ] || miss="$miss C-base_raw.csv"
  for f in solve_second_impl.py crosscheck_second_impl.py install_external_solver.sh; do
    [ -f "$R2/$f" ] || miss="$miss $f"
  done
  for f in "${!EXPECT[@]}"; do
    got=$(sha256sum "$WS/$f" 2>/dev/null | cut -c1-16)
    [ "$got" = "${EXPECT[$f]}" ] || miss="$miss ${f##*/}(sha ${got:-unreadable} != ${EXPECT[$f]})"
  done
  printf '%s' "$miss"
}

preflight_body() {
  require_venue
  OUT="$OUTROOT/$(date +%Y%m%dT%H%M%S)_second_impl"
  log "NAS=$NAS WS=$WS out=$OUT prefix=$PREFIX"
  log "solver python=$(PYBIN)"
  [ -d "$OUTROOT" ] && { log "OUTROOT already holds:"; ls -1 "$OUTROOT" | tail -8; }
  local miss; miss=$(check_blobs)
  [ -z "$miss" ] || log "MISSING/stale in this checkout:$miss"
  df -h "$NAS" | tail -1
  log "preflight fetched nothing, installed nothing, imported nothing"
  if [ "${WRITE_POINTER:-1}" = "1" ]; then
    mkdir -p "$OUT" && printf '%s\n' "$OUT" > "$OUTROOT/second_impl.POINTER" || die "cannot write pointer"
    log "pointer=$OUTROOT/second_impl.POINTER"
  fi
  seg_end "$([ -z "$miss" ] && echo DONE || echo PARTIAL)" "${miss:- none}"
  # the exit code has to carry it too: a caller that only reads rc=0 would treat
  # "preflight printed a MISSING line" as success.
  [ -z "$miss" ] || exit 1
}

install_body() {
  require_venue
  read_pointer
  [ -f "$R2/install_external_solver.sh" ] || die "$R2/install_external_solver.sh absent -- the pre-registered installer is not in this checkout"
  log "delegating to install_external_solver.sh (it owns the conda command, the two-attempt stop-loss,"
  log "the ${SEGMENT_S}s-then-report cap and the tar+sha256+restore line; this script does not re-declare them)"
  RUN="${RUN:-1}" CAP_MIN="${CAP_MIN:-60}" PREFIX="$PREFIX" \
    bash "$R2/install_external_solver.sh" 2>&1 | tail -25 || die "installer rc=$? -- report it; do not fall back to the laptop"
  "$(PYBIN)" -c "import dolfinx, sys; print('dolfinx', dolfinx.__version__, sys.version.split()[0])" \
    || die "dolfinx not importable from $PREFIX after the installer said it ran"
  seg_end DONE ""
}

smoke_body() {
  require_venue
  read_pointer
  local py; py=$(PYBIN); [ -x "$py" ] || die "no env python at $py -- install first"
  OUT="$OUT/smoke"; mkdir -p "$OUT" || die "cannot create $OUT"
  log "smoke = the REAL entry point on a 2x2 mesh (not a copy of its API calls)"
  timeout "$SEGMENT_S" "$py" "$R2/solve_second_impl.py" --nx 2 --ny 2 --out "$OUT" 2>&1 | tail -12 \
    || die "smoke solve failed rc=$? -- fix the code path before spending a real level"
  local rows; rows=$(( $(wc -l < "$OUT/second_impl_nodes.csv") - 1 ))
  log "SMOKE-GREEN rows=$rows (want (2+1)*(2+1)=9) -- this is NOT a cross-check verdict: a 2x2 mesh"
  log "is not converged, so its three integrals must not be quoted. It only proves the path runs."
  seg_end DONE ""
}

run_body() {
  require_venue
  read_pointer
  local py; py=$(PYBIN); [ -x "$py" ] || die "install first"
  OUT="$OUT/base"; mkdir -p "$OUT"
  [ -f "$R2/solve_second_impl.py" ] || die "solve_second_impl.py is not in this checkout -- no solve attempted"
  log "base level = the .edp's own border counts (nx=180 ny=40, .edp:53)"
  timeout "$SEGMENT_S" "$py" "$R2/solve_second_impl.py" --out "$OUT" 2>&1 | tail -10 \
    || die "base solve failed rc=$? -- read the log above; the frozen gate is untouched either way"
  log "crosscheck vs the shipped FreeFEM truth (this is the §4 row; FAIL is a legal terminal state)"
  timeout 300 "$py" "$R2/crosscheck_second_impl.py" --freefem "$REF_CSV" --other "$OUT/second_impl_nodes.csv" \
      --json "$OUT/crosscheck_base.json" 2>&1 | tail -8
  local rc=$?
  log "crosscheck rc=$rc  PASS=0 FAIL=1 (1% per quantity, three quantities judged separately)"
  seg_end "$([ $rc -eq 0 ] && echo DONE || echo RED)" "rc=$rc"
}

refine_body() {
  require_venue
  read_pointer
  local py; py=$(PYBIN); [ -x "$py" ] || die "install first"
  [ -f "$OUT/base/second_impl_nodes.csv" ] || die "no base level at $OUT/base -- run the base leg first"
  OUT="$OUT/refine"; mkdir -p "$OUT"
  log "doubled level = every buildmesh side doubled (.edp:53 180->360, 40->80)"
  timeout "$SEGMENT_S" "$py" "$R2/solve_second_impl.py" --nx 360 --ny 80 --out "$OUT" 2>&1 | tail -8 \
    || die "refined solve did not finish inside ${SEGMENT_S}s -- report, and §4 says report the two grids side by side"
  timeout 300 "$py" "$R2/crosscheck_second_impl.py" --freefem "$REF_CSV" --other "$OUT/second_impl_nodes.csv" \
      --json "$OUT/crosscheck_refine.json" 2>&1 | tail -8
  log "internal mesh gate: same implementation, two grids, three integrals, limit read from the tool"
  timeout 300 "$py" "$R2/crosscheck_second_impl.py" --freefem "$OUT/../base/second_impl_nodes.csv" \
      --other "$OUT/second_impl_nodes.csv" --json "$OUT/mesh_internal.json" > "$OUT/mesh_internal.stdout" 2>&1
  "$py" - "$OUT" "$R2" <<'PY' || die "could not read the mesh gate's own numbers"
import json, sys
sys.path.insert(0, sys.argv[2])
import crosscheck_second_impl as x
d = json.load(open(sys.argv[1] + "/mesh_internal.json", encoding="utf-8"))
lim = x.MESH_REL_LIMIT
for k in x.QUANTITIES:
    r = d["quantities"][k]["rel_diff"]
    print(f"  mesh-change {k:2s} rel={r:.4%} limit={lim:.0%} {'OK' if r <= lim else 'OVER'}")
print(("MESH-GATE PASS" if all(d["quantities"][k]["rel_diff"] <= lim for k in x.QUANTITIES)
       else "MESH-GATE OVER -- section 1 says report BOTH grids, not only the finer one"))
PY
  seg_end DONE ""
}

status_body() {
  log "read-only: venue, pointer, env, tree. Nothing installed, nothing solved."
  local why miss=""
  if ! why=$(venue_check); then
    log "off-venue: $why -> every other mode refuses by design"
    seg_end PARTIAL "off-venue"
    exit 0
  fi
  log "python=$(PYBIN) prefix-present=$([ -x "$PREFIX/bin/python" ] && echo yes || echo no)"
  if [ -f "$OUTROOT/second_impl.POINTER" ]; then
    log "pointer=$(head -1 "$OUTROOT/second_impl.POINTER")"
  else
    log "pointer=ABSENT -> run preflight"; miss=" pointer"
  fi
  miss="$miss $(check_blobs)"
  [ -n "${miss// /}" ] && log "MISSING/stale:$miss"
  seg_end "$([ -z "${miss// /}" ] && echo DONE || echo PARTIAL)" "${miss:- none}"
  # the exit code carries it too: a caller that reads only rc would otherwise treat "preflight
  # printed a MISSING line" as success.
  [ -z "${miss// /}" ] || exit 1
}

case "$MODE" in
  preflight) preflight_body ;;
  install)   install_body ;;
  smoke)     smoke_body ;;
  run)       run_body ;;
  refine)    refine_body ;;
  status)    status_body ;;
  *) die "unknown mode '$MODE' (preflight|install|smoke|run|refine|status)" ;;
esac
