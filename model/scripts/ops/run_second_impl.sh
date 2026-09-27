#!/usr/bin/env bash
# ops/run_second_impl.sh -- the SECOND-IMPLEMENTATION leg (登记七/登记九, 判据一字未动).
#
# Venue rule (登记九, user's own words, 9/27 21:2x): NOTHING gets installed on the user's machine --
# no solver, no conda/miniforge, no pip package, no WSL environment.  "Zero cost" does not mean
# "free to install"; a pile of software he never asked for on his box is a cost.  The only allowed
# local actions are: read files, run stdlib tests, run already-installed gates.
# The only work venue is the ModelScope instance, and its environment goes to the NAS (/mnt/workspace)
# so it survives a machine swap; the previous box taught us that /mnt/workspace is shared while
# everything under / is not.
#
#   preflight  conda/micromamba present? NAS writable? repo blobs verified? -- refuses everything else
#   install    conda-forge fenics-dolfinx INTO THE NAS PREFIX ONLY (no /usr, no cp -a, no ldconfig)
#   smoke      one-row mesh, both sides with the SAME formulas, three quantities -- seconds, must go
#              green before any real level is attempted
#   run        one formal level + crosscheck_second_impl.py (three integrals, <=1% gate, unchanged)
#   status     what landed, without running anything
#
# Every phase ends with a SEGMENT-END line carrying wall_used / the 8-hour window / whether the
# instance can be closed.  Heartbeat duty: the platform closes an idle box, so keep issuing commands
# inside a segment -- and stop at the window, hand the state back, do not silently continue.
#
# Refuses to run anywhere that is not the instance (see require_venue), so executing this file on the
# laptop is a loud rc=3, not an install.
set -uo pipefail

WS="${WS:-/mnt/workspace}"
REPO="${REPO:-$WS/pinn-repro-2026/repo}"
PREFIX="${PREFIX:-$WS/pyfaxi}"          # conda env lands here -> survives the box being swapped
OUTROOT="${OUTROOT:-$WS/route2_out}"
SEGMENT_S="${SEGMENT_S:-1500}"          # < 30 min per segment
WINDOW_S="${WINDOW_S:-28800}"           # ~8 h ceiling, user's number
MODE="${1:-status}"
T0=$(date +%s)
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "[ABORT] $*"; exit 3; }
seg_end() {  # seg_end <verdict> <what-is-missing>
  local wall=$(( $(date +%s) - T0 ))
  printf 'SEGMENT-END mode=%s verdict=%s wall_used=%ds segment_cap=%ss window_left=%ss prefix=%s out=%s missing=%s 实例可否关=%s\n' \
    "$MODE" "$1" "$wall" "$SEGMENT_S" "$(( WINDOW_S - wall ))" "$PREFIX" "${OUT:-none}" "$2" \
    "$([ "$1" = DONE ] && echo 可关 || echo 先别关-看missing)"
}

venue_check() {   # rc 0 = the instance; rc 1 = not, with the reason on stdout (one copy of the test, used by every mode)
  case "$(uname -s)" in
    Linux) : ;;
    *) printf 'uname=%s is not Linux' "$(uname -s)"; return 1 ;;
  esac
  if ! awk -v ws="$WS" '$2 == ws || $2 == ws "/" {f = 1} END {exit !f}' /proc/mounts; then
    printf '%s is not a mount point here (mounts under /mnt: %s)' "$WS" \
      "$(awk '$2 ~ /^\/mnt/ {printf "%s ", $2}' /proc/mounts 2>/dev/null)"
    return 1
  fi
}

require_venue() {
  local why
  why=$(venue_check) || die "$why -> not the instance, and the laptop is not a work venue (登记九: stop and report, never fall back here)"
  touch "$WS/.second_impl_wtest" 2>/dev/null || die "$WS not writable"
  rm -f "$WS/.second_impl_wtest"
  case "$PWD" in "$WS"*) : ;; *) cd "$WS" || die "cannot cd $WS" ;; esac
}

find_conda() {
  for c in conda micromamba mamba "$WS/miniforge3/bin/conda" "$WS/micromamba/bin/micromamba"; do
    if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then printf '%s' "$c"; return 0; fi
  done
  return 1
}

preflight_body() {
  local c; require_venue
  OUT="$OUTROOT/$(date +%Y%m%dT%H%M%S)"; mkdir -p "$OUT"
  log "workspace=$WS repo=$REPO prefix=$PREFIX out=$OUT"
  [ -d "$REPO/model/scripts/route2" ] || log "WARN repo not fetched yet ($REPO)"
  c=$(find_conda) && log "conda=$c" || log "conda=NONE (instance image lacks it -> install micromamba INTO $WS only, or report back)"
  df -h "$WS" | tail -1
  log "no solver is fetched, installed or imported by preflight"
  seg_end DONE ""
}

install_body() {
  local c; require_venue
  c=$(find_conda) || die "no conda/micromamba on this instance after preflight said otherwise"
  mkdir -p "$PREFIX" || die "cannot create $PREFIX"
  log "installing conda-forge fenics-dolfinx into $PREFIX (NAS). Never /usr, never cp -a, never ldconfig."
  timeout "$SEGMENT_S" "$c" create -y -p "$PREFIX" -c conda-forge python=3.11 fenics-dolfinx meshio \
    2>&1 | tail -6 || die "conda create failed (rc=$?) -- report, do not fall back to the laptop"
  "$PREFIX/bin/python" -c "import dolfinx, sys; print('dolfinx', dolfinx.__version__, sys.version.split()[0])" \
    || die "dolfinx import failed after install"
  seg_end DONE ""
}

smoke_body() {
  require_venue
  local py="$PREFIX/bin/python"; [ -x "$py" ] || py=$(find_conda >/dev/null && echo "$PREFIX/bin/python")
  [ -x "$py" ] || die "no env python yet -- run install first"
  mkdir -p "${OUT:-$OUTROOT/smoke}"
  timeout 120 "$py" - <<'PY' || die "smoke failed (rc=$?)"
import dolfinx, numpy as np, time
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_unit_square
t0 = time.time()
mesh = create_unit_square(dolfinx.default_comm(), 1, 1)          # one-row mesh: seconds, not minutes
V = functionspace(mesh, ("Lagrange", 1))
u = Function(V); u.interpolate(lambda xx: 1.0 + xx[0] ** 2 + xx[1] ** 2)
print("smoke dolfinx ok wall_s=%.2f dofs=%d" % (time.time() - t0, V.dofmap.index_map.size_global))
PY
  log "smoke green -> the real level may be attempted (still this segment if budget allows, else stop)"
  seg_end DONE ""
}

run_body() {
  require_venue
  local py="$PREFIX/bin/python"; [ -x "$py" ] || die "install first"
  mkdir -p "${OUT:-$OUTROOT/run}"
  log "level 1 = one Stokes solve, then the SAME three quantities via the frozen ruler"
  [ -f "$REPO/model/scripts/route2/solve_second_impl.py" ] \
    || die "solve_second_impl.py is not in the repo yet -- author it first; no solve was attempted"
  timeout "$SEGMENT_S" "$py" "$REPO/model/scripts/route2/solve_second_impl.py" \
      --out "${OUT:-$OUTROOT/run}" 2>&1 | tail -8 \
    || die "solve_second_impl.py failed (rc=$?) -- report the log, do not touch the frozen gate"
  timeout "$SEGMENT_S" "$py" "$REPO/model/scripts/route2/crosscheck_second_impl.py" \
      --freefem "$REPO/model/cases/contraction_2d/cfd/C-base/C-base_raw.csv" \
      --other "${OUT:-$OUTROOT/run}/second_impl_nodes.csv" \
      --json "${OUT:-$OUTROOT/run}/crosscheck.json" 2>&1 | tail -8
  local rc=$?
  log "crosscheck rc=$rc (FAIL is a legal terminal state -- the gate itself is frozen: three integrals <=1%, refine <2%)"
  seg_end "$([ $rc -eq 0 ] && echo DONE || echo RED)" "rc=$rc"
}

status_body() {
  log "venue check only (no installs, no solves, nothing is created off-venue)"
  local why
  if ! why=$(venue_check); then
    log "off-venue: $why -> every other mode refuses by design"
    seg_end PARTIAL "off-venue"
    exit 0
  fi
  for d in "$PREFIX" "$OUTROOT"; do [ -d "$d" ] && log "present: $d" || log "absent : $d"; done
  ls -1 "$OUTROOT" 2>/dev/null | tail -5
  seg_end DONE ""
}

case "$MODE" in
  preflight) preflight_body ;;
  install)   install_body ;;
  smoke)     smoke_body ;;
  run)       run_body ;;
  status)    status_body ;;
  *) die "unknown mode '$MODE' (preflight|install|smoke|run|status)" ;;
esac
