#!/usr/bin/env bash
# Second-implementation solver install + ONE-SHOT PERSISTENCE, for the 2026-09-27
# cross-validation work order (paper-route2/第二实现交叉验证-预注册-20260927.md).
#
# WRITTEN, NOT RUN. This line does not touch the instance; the 统括官 runs it.
# Default mode is a dry run that prints what it would do; set RUN=1 to execute.
#
# Two rules it exists to honour, both learned the expensive way in this project:
#   1. Everything lands under the NAS workspace ($PREFIX), never in a throwaway env, and the
#      script ENDS by packaging that tree into a tarball + sha256 + a one-line restore
#      command. A user who has to reinstall after an instance restart did not get a tool.
#   2. Hard stop-loss: FEniCSx first (conda-forge, then pip wheel -- ONE retry each),
#      OpenFOAM only from a PREBUILT conda package, and NEVER from source. Total wall clock
#      is capped; past the cap the script reports what it got and the last evidence line,
#      and the work order's INDETERMINATE row applies. No GPU, no money.
set -uo pipefail

PREFIX="${PREFIX:-/mnt/workspace/pinn-repro-2026/ext-fenics}"
ARCHIVE="${ARCHIVE:-/mnt/workspace/pinn-repro-2026/extsolver.tgz}"
CAP_MIN="${CAP_MIN:-60}"                       # pre-registered cap: installation wall clock
START_EPOCH="$(date +%s)"
RUN="${RUN:-0}"
LOG="${LOG:-extsolver_install_$(date +%Y%m%dT%H%M%S).log}"

log() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"; }
budget_left_min() { echo $(( (CAP_MIN * 60 - ($(date +%s) - START_EPOCH)) / 60 )); }

# Every command goes through here, so the dry run is the same script as the real one rather
# than a parallel copy that can drift.
do_() {
  if [ "$RUN" = "1" ]; then log "RUN  $*"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}";
  else log "PLAN $*"; return 0; fi
}

log "mode=$RUN (RUN=1 to execute); prefix=$PREFIX; cap=${CAP_MIN}min; log=$LOG"
[ "$RUN" = "1" ] || { log "dry run only -- nothing was created"; }

# --- step 0: probe the channel once, do not retry-loop (work order section 0) -----------
probe() {
  local url="$1" code
  code="$(curl -s -o /dev/null -m 20 -r 0-0 -w '%{http_code}' "$url" || echo 000)"
  log "probe $url -> HTTP $code"
  case "$code" in 2*|3*) return 0 ;; *) return 1 ;; esac
}
# repodata.json is literally what conda fetches, so a hit here means the channel is usable;
# a 404 would NOT mean unreachable (measured 2026-09-27: /channelinfo gives 404 while
# conda-forge/noarch/repodata.json gives 206 with a range request).
probe "https://conda.anaconda.org/conda-forge/noarch/repodata.json" || {
  echo "UNREACHABLE: conda-forge not reachable; stop and report this line (INDETERMINATE row)"
  exit 3
}

mkdir_ok=1
if [ "$RUN" = "1" ]; then mkdir -p "$PREFIX" || mkdir_ok=0; fi
[ "$mkdir_ok" = "1" ] || { echo "cannot create $PREFIX -- refusing to install into a temp env"; exit 3; }

# --- step 1: FEniCSx / dolfinx ----------------------------------------------------------
# conda-forge is the ONLY route this script takes: measured 2026-09-27,
# `pypi.org/simple/fenics-dolfinx/` returns 404, so the "pip wheel" fallback in the work
# order is not available under that name -- recorded rather than assumed. If conda is absent,
# the script reports and stops instead of improvising an install into a temp env.
FENICS_OK=0
if command -v mamba >/dev/null 2>&1 || command -v conda >/dev/null 2>&1; then
  PM="$(command -v mamba || command -v conda)"
  do_ "$PM" create -y -p "$PREFIX" -c conda-forge python=3.11 fenics-dolfinx=0.9 petsc4py \
        numpy h5py meshio gmsh && FENICS_OK=1
  if [ "$FENICS_OK" = "0" ]; then
    # ONE retry, different lever: drop the pin, keep the channel. A second retry is out of budget.
    log "fenics attempt 1 failed; ONE retry without version pins (budget $(budget_left_min) min left)"
    do_ "$PM" create -y -p "$PREFIX" -c conda-forge python=3.11 fenics-dolfinx \
          numpy h5py meshio gmsh && FENICS_OK=1
  fi
else
  log "no conda/mamba on PATH -- skipping the conda route (report, do not hand-install)"
fi
[ "$FENICS_OK" = "1" ] || log "FEniCSx NOT installed within its two attempts; going to step 2"

# --- step 2: OpenFOAM, prebuilt only ----------------------------------------------------
FOAM_OK=0
if [ "$(budget_left_min)" -ge 25 ]; then
  PM="$(command -v mamba || command -v conda || true)"
  if [ -n "$PM" ]; then
    do_ "$PM" search -c conda-forge openfoam >/dev/null 2>&1 && {
      log "prebuilt openfoam package exists; installing (NO source compile, ever)"
      do_ "$PM" create -y -p "$PREFIX-of" -c conda-forge openfoam && FOAM_OK=1
    } || log "no prebuilt openfoam on conda-forge for this platform -> give up, write UNREACHABLE"
  fi
else
  log "budget left $(budget_left_min) min < 25 -> OpenFOAM step skipped by the cap"
fi
[ "$FOAM_OK" = "1" ] || log "OpenFOAM not installed (prebuilt-only rule honoured)"

# --- step 3: verify presence ISN'T usability, then persist ------------------------------
# The lesson from the FreeFEM archive: a package can be present and still not run. But the
# other lesson (this project, 9/26) is that a probe written against an API nobody has run
# fails in a way that looks like a solver failure. So the split is deliberate:
#   * `import dolfinx` failing        -> real: PRESENT-BUT-UNUSABLE, exit 4.
#   * the optional mini-snippet failing -> NOT judged: it prints VERIFY_PROBE_ERROR and the
#     traceback for a human, because these calls are unverified guesses about the API.
verify() {
  [ "$RUN" = "1" ] || { log "PLAN verify (dry run)"; return 0; }
  local py="$PREFIX/bin/python"
  [ -x "$py" ] || py="$(command -v python3)"
  "$py" - <<'PY' 2>&1 | tee -a "$LOG"
import traceback
try:
    import dolfinx
    print("dolfinx import OK:", dolfinx.__version__)
except Exception as exc:                                     # noqa: BLE001
    print("DOLFINX PRESENT-BUT-UNUSABLE:", type(exc).__name__, exc)
    raise SystemExit(4)
try:                                                         # optional, NOT judged
    from dolfinx import fem, mesh
    m = mesh.create_unit_square(__import__("mpi4py").MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(m, ("Lagrange", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: x[0] ** 2)
    print("dolfinx mini-surface OK: dofs =", u.vector.size)
except Exception:                                            # noqa: BLE001
    print("VERIFY_PROBE_ERROR (probe defect, not a solver verdict) -- human, read this:")
    traceback.print_exc()
PY
}
VERIFY_RC=0
[ "$FENICS_OK" = "1" ] && { verify || VERIFY_RC=4; }

# --- the persistence tail: this is the deliverable, not an afterthought -----------------
persist() {
  [ "$RUN" = "1" ] || { log "PLAN persist"; return 0; }
  tar czf "$ARCHIVE" -C "$(dirname "$PREFIX")" "$(basename "$PREFIX")" 2> >(tee -a "$LOG" >&2) || {
    echo "tar failed -- the environment is NOT persisted, do not close the instance yet"; return 5; }
  ( cd "$(dirname "$ARCHIVE")" && sha256sum "$(basename "$ARCHIVE")" > "$ARCHIVE.sha256" )
  tee -a "$LOG" <<EOF

=== RECOVERY (one line, from a fresh instance, after mounting the same NAS) ===
  tar xzf $ARCHIVE -C / && sha256sum -c $ARCHIVE.sha256
  then re-run this script's verify() step -- presence is not usability:
  the acceptance is 'import dolfinx + interpolate + a mini solve', not 'ls'.
================================================================================
EOF
  log "archive: $ARCHIVE ($(stat -c %s "$ARCHIVE" 2>/dev/null || stat -f %z "$ARCHIVE") bytes)"
  sha256sum "$ARCHIVE" | tee -a "$LOG"
}
persist

ELAPSED=$(( $(date +%s) - START_EPOCH ))
log "elapsed ${ELAPSED}s of $((CAP_MIN * 60))s cap; fenics_ok=$FENICS_OK foam_ok=$FOAM_OK verify_rc=$VERIFY_RC"
if [ "$RUN" != "1" ]; then
  echo "OUTCOME: DRY RUN -- nothing was attempted, so no outcome is judged. Re-run with RUN=1 on the instance."
  exit 0
fi
if [ "$FENICS_OK" = "0" ] && [ "$FOAM_OK" = "0" ]; then
  echo "OUTCOME: no second implementation available -> the pre-registration's INDETERMINATE row applies. Report this line, do not write 'passed' or 'failed'."
  exit 6
fi
exit 0
