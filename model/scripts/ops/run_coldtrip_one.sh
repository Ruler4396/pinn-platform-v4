#!/usr/bin/env bash
# ops/run_coldtrip_one.sh -- ONE machine trip, THREE readings, on a COLD instance.
#
# Written 9/27 14:2x after dsw-2213486 turned out to have been REPLACED (new: dsw-v5n5egmoh4wwjrx2se,
# ModelScope id 2213920) rather than merely sick, so nothing is assumed present: not FreeFEM, not the
# NAS copy of the last trip, not a writable disk.  The earlier "notebook process not listening" reading
# was an over-inference from the same 404s; the replacement explains them.
#
#   preflight   what exists; refuses to let anything downstream run without a resolvable solver
#   restore     unpack ffroot.tgz (only useful if preflight found the archive and no binary)
#   p1          R2-7 constraints 2+3: four Re levels solved, |p_star|max ratio MEASURED
#   p2          K0b: syntax probe -> 12-digit staged emission -> reference scan (drives the
#               hash-pinned ops/run_k0b_5236655.sh, which refuses a 6-digit reference)
#   p3          E5 CFD unit cost -- a thesis-side cell (bi-zuo-4 tail) measured on route-2 machine
#               time, with the three guards: solver probe first, rc!=0 excluded, >=5/7 to land
#   status      what landed, without running anything
#
# Segment discipline: each public phase runs its body as a CHILD (`bash $0 <phase>_body`) under
# `timeout $SEGMENT_S` (default 1500 s < the 30-min cap).  The body streams one line per step; the
# parent always prints SEGMENT-END with used-wall / can-shut-down / what-is-missing, including when
# the child is cut.  Phases are independent, so an idle shutdown costs at most the segment in flight.
# No training, no money, no rewrite of pushed history.
set -uo pipefail

FULL_PIN="${FULL_PIN:-549fa3a7720c54fed92e1324e43537670b3e5508}"   # the commit the WANT table describes
REPO=Ruler4396/pinn-platform-v4
WS="${WS:-/mnt/workspace/pinn-repro-2026}"
FFROOT="${FFROOT:-$WS/ffroot.tgz}"
OUTD="${OUTD:-$WS/out/coldtrip_20260927}"
SEGMENT_S="${SEGMENT_S:-1500}"
NS_LEVELS="${NS_LEVELS:-1e-3 1 10 50}"
NS_COUNT=$(set -- $NS_LEVELS; echo $#)
E5_ATTEMPTS="${E5_ATTEMPTS:-7}"
E5_MIN_OK="${E5_MIN_OK:-5}"
MODE="${1:-preflight}"
T0=$(date +%s)
INSTANCE="${HOSTNAME:-unknown-host}"
[ -r /etc/machine-id ] && INSTANCE="$INSTANCE+$(cut -c1-8 /etc/machine-id)"

say() { printf '%s | %s\n' "$(date '+%H:%M:%S')" "$*"; }
used() { echo $(( $(date +%s) - T0 )); }
seg_end() { # verdict  shutdown-advice  what-is-missing
  say "SEGMENT-END phase=${TARGET:-$MODE} verdict=$1 wall_used=$(used)s segment_cap=${SEGMENT_S}s instance=$INSTANCE"
  say "SEGMENT-END shutdown=$2 missing=$3"
}

# sha256(blob bytes) prefix, measured at the pin below this turn (all 13 re-read at 4d76014,
# which is where this driver itself lives -- the earlier lesson: a table partly measured at an
# older pin makes the instance side FETCH badhash and burns the trip).
declare -A WANT=(
  [model/scripts/gen_ns_re_edp.py]=530a74544dc046d7
  [model/scripts/finalize_ns_truth.py]=adc427cf87bc8995
  [model/scripts/check_ns_re_to_stokes.py]=912f17e0a8088ff0
  [model/scripts/route2/generate_t_case.py]=b31bf371cddeb444
  [model/scripts/route2/k0_truth_gate.py]=bdfdc971e65c2ebe
  [model/scripts/route2/artifacts.py]=1febd1e1dc89ae66
  [model/scripts/route2/residual_scorers.py]=86c96b1cbf9c6bce
  [model/scripts/route2/t_geometry.py]=94329e67f178b7df
  [model/scripts/route2/selftest_route2_stdlib.py]=73ebf7eb0b7aa688
  [model/scripts/ops/run_k0b_5236655.sh]=3d5a0662940e1753
  [model/cases/contraction_2d/cfd/C-base/C-base_stokes.edp]=2a62e0d41aa2fe98
  [model/cases/contraction_2d/cfd/C-base/C-base_raw.csv]=46bd0401cf0f92f5
  [model/cases/contraction_2d/cfd/C-base_ns_re1/probe_syntax.edp]=a4ca809f0b05b932
)
declare -A BYTES=(
  [model/cases/contraction_2d/cfd/C-base/C-base_stokes.edp]=2352
  [model/cases/contraction_2d/cfd/C-base/C-base_raw.csv]=99976
)

fetch() { # path -> $WS/<path>, refused unless sha256 prefix matches WANT
  local p="$1" tmp got want wantb m
  want="${WANT[$p]:-}"; wantb="${BYTES[$p]:-}"
  tmp="$OUTD/$(basename "$p").part"
  mkdir -p "$(dirname "$WS/$p")" "$OUTD" || return 1
  for m in "https://gh-proxy.com/https://raw.githubusercontent.com" "https://raw.githubusercontent.com"; do
    if curl -fsSL --max-time 90 --retry 2 -o "$tmp" "$m/$REPO/$FULL_PIN/$p" 2>/dev/null; then
      got=$(sha256sum "$tmp" | cut -c1-16)
      if [ -n "$want" ] && [ "$got" != "$want" ]; then
        say "FETCH $(basename "$p") REFUSED: sha256=$got != pinned $want (wrong pin or bad transfer)"
        rm -f "$tmp"; return 1
      fi
      if [ -n "$wantb" ] && [ "$(stat -c%s "$tmp")" != "$wantb" ]; then
        say "FETCH $(basename "$p") REFUSED: $(stat -c%s "$tmp") B != pinned $wantb B"
        rm -f "$tmp"; return 1
      fi
      mv "$tmp" "$WS/$p"
      say "FETCH ok $(basename "$p") sha256=$got bytes=$(stat -c%s "$WS/$p") via ${m#https://}"
      return 0
    fi
  done
  say "FETCH $(basename "$p") FAILED on both mirrors"; rm -f "$tmp"; return 1
}

FFHOME="${FFHOME:-$WS/ffrun}"
DEB_DIR="${DEB_DIR:-$WS/debs}"
# The four sonames FreeFEM's `ldd` demands are NOT inside ffroot.tgz (that archive was built from
# `dpkg -L` + `ldd` on the earlier box, and the packages were apt-installed afterwards), and their
# entries in the archive are dangling relative symlinks.  Measured on dsw-2213920 9/27 14:29-14:31:
# unpacking these jammy packages into $FFHOME with `dpkg-deb -x` and exporting LD_LIBRARY_PATH gets
# `ldd` to `not found: 0` -- no /usr writes, no ldconfig, nothing installed system-wide.
FFLIB_PKGS="${FFLIB_PKGS:-libumfpack5 libcholmod3 libarpack2 libhdf5-103-1 libamd2 libcamd2 libccolamd2 libcolamd2 libsuitesparseconfig5 libmetis5 libsz2 libaec0}"
# HARD CONSTRAINT (orchestrator, 9/27 14:2x): recovery lands ONLY in an instance-local directory plus
# LD_LIBRARY_PATH.  No `cp -a` into /usr, no `ldconfig`, no extraction to `/` -- the previous instance
# was replaced after exactly that kind of recovery, so if the archive cannot satisfy the loader path
# this script reports the missing list and stops instead of touching system directories.
FFBIN=""

ff_libdirs() { # every directory under the unpacked prefix that holds a shared object
  find "$FFHOME" -name '*.so*' -type f 2>/dev/null | sed 's|/[^/]*$||' | sort -u | tr '\n' ':'
}

export_fflib() { # prepend them to LD_LIBRARY_PATH for this process and its children
  local d; d=$(ff_libdirs)
  [ -n "$d" ] && export LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:-}"
  printf '%s' "$d"
}

solver_path() { # print an executable to use: PATH first, then the instance-local unpacked copy.
                # v4.9's binary is FreeFem++ (capital F) -- probing one lowercase name once produced
                # a false "not installed" on a box that had it, so all four spellings are tried.
  local c
  for c in FreeFem++ freefem++ FreeFem freefem; do
    command -v "$c" 2>/dev/null && return 0
  done
  local cand
  for cand in "$FFHOME/usr/bin/FreeFem++" "$FFHOME/usr/bin/freefem++" "$FFHOME/bin/FreeFem++"; do
    [ -x "$cand" ] && { echo "$cand"; return 0; }
  done
  find "$FFHOME" -type f -name 'FreeFem++' 2>/dev/null | head -1
}

require_solver() { # re-resolve inside the child process that actually solves
  local ff libd
  if ff=$(solver_path); then
    FFBIN="$ff"
    case "$ff" in
      "$FFHOME"*) export_fflib >/dev/null; say "SOLVER unpacked copy $ff LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" ;;
      *)          say "SOLVER on PATH: $ff";;
    esac
    local miss; miss=$(ldd "$ff" 2>/dev/null | awk '/not found/{print $1}' | tr '\n' ' ')
    if [ -n "$miss" ]; then
      say "SOLVER REFUSE: unresolved libraries [$miss] -- reporting, NOT installing into /usr"
      seg_end REFUSED "yes" "loader path for: $miss"
      return 1
    fi
    return 0
  fi
  say "SOLVER REFUSE: nothing on PATH and nothing under $FFHOME -- run '$0 restore' then '$0 preflight'"
  seg_end REFUSED "yes" "a solver resolvable without touching system directories"
  return 1
}

require_preflight() { # phases refuse unless preflight PASSED for THIS pin -- the silent path is
                      # the one that records rc=127 as a wall time and calls the trip a success
  local mark="$OUTD/.preflight.pass"
  if [ ! -f "$mark" ]; then
    say "$( [ -n "${TARGET:-}" ] && echo "$TARGET" ) REFUSE: preflight has not passed on this instance (no $mark). Run '$0 preflight' first."
    return 1
  fi
  if [ "$(cat "$mark")" != "$FULL_PIN" ]; then
    say "REFUSE: preflight passed for pin $(cat "$mark"), not $FULL_PIN -- re-run preflight after any environment change"
    return 1
  fi
  return 0
}

# ------------------------------------------------------------------ preflight
do_preflight() {
  local fail=0 ff miss avail probe
  say "PREFLIGHT instance=$INSTANCE date='$(date '+%Y-%m-%d %H:%M:%S %z')' pin=${FULL_PIN:0:7}"
  say "PREFLIGHT python: $(python3 -V 2>&1 || echo MISSING)"
  python3 -c 'import json,math,csv,statistics' 2>/dev/null \
    && say "PREFLIGHT stdlib json/math/csv/statistics: ok" \
    || { say "PREFLIGHT REFUSE: stdlib imports broken -- the gates themselves cannot run"; fail=1; }

  if [ -d "$WS" ]; then
    say "PREFLIGHT workspace $WS: $(ls -1 "$WS" 2>/dev/null | wc -l) entries"
    if [ -f "$WS/model/cases/contraction_2d/cfd/C-base/C-base_raw.csv" ]; then
      say "PREFLIGHT NAS cross-instance sharing: the previous tree IS visible (still re-fetched and hash-checked)"
    else
      say "PREFLIGHT NAS cross-instance sharing: previous artefacts NOT visible -> cold instance, everything fetched by hash"
    fi
  else
    say "PREFLIGHT workspace $WS not mounted -> falling back to /tmp/pinn-coldtrip; nothing from the last trip is assumed"
    WS=/tmp/pinn-coldtrip; OUTD="$WS/out"; mkdir -p "$OUTD" || { say "PREFLIGHT REFUSE: cannot create $OUTD"; fail=1; }
  fi
  if [ -f "$FFROOT" ]; then
    say "PREFLIGHT ffroot archive: $(stat -c%s "$FFROOT") B sha256=$(sha256sum "$FFROOT" | cut -c1-16)"
  else
    say "PREFLIGHT ffroot archive ABSENT ($FFROOT) -- 'restore' cannot help here; FreeFEM would need reinstalling"
  fi

  probe="$OUTD/.write_probe"
  if echo x > "$probe" 2>/dev/null; then say "PREFLIGHT disk writable: yes ($OUTD)"; rm -f "$probe"
  else say "PREFLIGHT REFUSE: $OUTD not writable"; fail=1; fi
  avail=$(df -Pk "$OUTD" 2>/dev/null | awk 'NR==2{print $4}')
  say "PREFLIGHT free space on that fs: ${avail:-unknown} KB"
  [ -n "${avail:-}" ] && [ "$avail" -lt 204800 ] && { say "PREFLIGHT REFUSE: < 200 MB free"; fail=1; }

  if ff=$(solver_path); then
    FFBIN="$ff"; case "$ff" in "$FFHOME"*) export_fflib >/dev/null;; esac
    say "PREFLIGHT solver: $FFBIN LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
    miss=$(ldd "$ff" 2>/dev/null | awk '/not found/{print $1}' | tr '\n' ' ')
    if [ -n "$miss" ]; then
      say "PREFLIGHT REFUSE: solver present but unresolved libs: $miss -- an unloadable solver is exactly what an rc=127 / 'wall=1 ms' record looks like"
      fail=1
    else
      say "PREFLIGHT solver ldd: all shared objects resolved"
    fi
  else
    say "PREFLIGHT REFUSE: no solver on PATH nor under $FFHOME (probed FreeFem++ / freefem++ / FreeFem / freefem)"
    # The marker means "a solver will actually run", not "an archive exists": measured on the
    # instance 9/27 14:25, the archive-present branch used to print a next step and still PASS,
    # which handed p1/p2/p3 a green light with nothing to solve with.
    [ -f "$FFROOT" ] && say "  next: '$0 restore' unpacks it into $FFHOME and, if the archive lacks the sonames, fetches them into the same prefix (no /usr writes), then preflight again"
    [ -f "$FFROOT" ] || say "  and $FFROOT is absent too, so FreeFEM would have to be installed"
    fail=1
  fi

  if curl -fsS -m 25 -o /dev/null "https://raw.githubusercontent.com/$REPO" 2>/dev/null; then
    say "PREFLIGHT github reachable: yes"
  else
    say "PREFLIGHT github probe failed (mirror window or DNS); fetch() tries gh-proxy first and prints which mirror served each file"
  fi

  if [ "$fail" = 0 ]; then
    echo "$FULL_PIN" > "$OUTD/.preflight.pass"
    say "PREFLIGHT RESULT: PASS (marker $OUTD/.preflight.pass = ${FULL_PIN:0:7}) -- run p1, p2, p3 as separate segments"
    seg_end PASS "no, work remains" "nothing (preflight only)"
  else
    rm -f "$OUTD/.preflight.pass"
    say "PREFLIGHT RESULT: REFUSED -- marker cleared; p1/p2/p3 will now refuse until it passes"
    seg_end REFUSED "yes, nothing was measured" "see the REFUSE lines above"
  fi
  return "$fail"
}

fetch_libs() { # download the .debs and unpack them into the instance-local prefix ONLY
  mkdir -p "$DEB_DIR" || { say "LIBS REFUSE: cannot create $DEB_DIR"; return 1; }
  say "LIBS apt-get download (jammy, mirrors as configured on this box): $FFLIB_PKGS"
  ( cd "$DEB_DIR" && apt-get download $FFLIB_PKGS ) >"$OUTD/libs_apt.txt" 2>&1 \
    || { say "LIBS REFUSE: apt-get download failed (see $OUTD/libs_apt.txt); will NOT touch /usr"; return 1; }
  local n=0 d
  for d in "$DEB_DIR"/*.deb; do
    [ -f "$d" ] || continue
    dpkg-deb -x "$d" "$FFHOME" || say "LIBS note: dpkg-deb -x $d exited $?"
    n=$((n + 1))
  done
  say "LIBS unpacked $n package trees into $FFHOME"
  export_fflib >/dev/null
  say "LIBS LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
}
do_restore() {
  [ -f "$FFROOT" ] || { say "RESTORE REFUSE: no $FFROOT on this instance"; return 1; }
  mkdir -p "$FFHOME" || { say "RESTORE REFUSE: cannot create $FFHOME"; return 1; }
  say "RESTORE unpacking $FFROOT into $FFHOME (instance-local prefix only: no /usr writes, no ldconfig)"
  tar xzf "$FFROOT" -C "$FFHOME" || say "RESTORE note: tar exited nonzero; checking what did land"
  say "RESTORE landed dirs: $(find "$FFHOME" -maxdepth 3 -type d 2>/dev/null | head -8 | tr '\n' ' ')"
  local ff miss
  export_fflib >/dev/null   # NOT $(export_fflib): command substitution exports into a subshell only,
  say "RESTORE LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>} (this shell and its children only)"
  # and the ldd below then reports the archive's own libraries as missing -- measured on the
  # instance 9/27 14:26, where libumfpack/libcholmod/libarpack/libhdf5 were all present in ffrun/lib
  if ff=$(solver_path); then
    miss=$(ldd "$ff" 2>/dev/null | awk '/not found/{print $1}' | tr '\n' ' ')
    say "RESTORE candidate: $ff unresolved_libs=[$miss]"
      if [ ! -f "$OUTD/.libs.done" ]; then
        say "RESTORE attempting the documented package recovery into $FFHOME (instance-local, no /usr)"
        mkdir -p "$OUTD" && touch "$OUTD/.libs.done"
        fetch_libs
        miss=$(ldd "$ff" 2>/dev/null | awk '/not found/{print $1}' | tr '\n' ' ')
        if [ -z "$miss" ]; then
          say "RESTORE ok after fetch_libs: $ff -- all shared objects resolved"
          return 0
        fi
        say "RESTORE still unresolved after recovery: [$miss]"
      fi
    if [ -n "$miss" ]; then
      say "RESTORE INCOMPLETE: the archive does not carry those objects. Per the standing constraint this run will NOT copy them into /usr or run ldconfig -- the missing list is written to $OUTD/missing_libs.txt and escalated."
      echo "$miss" > "$OUTD/missing_libs.txt"
      return 2
    fi
    say "RESTORE ok -- rerun '$0 preflight' to re-mark the environment as ready"
    return 0
  fi
  say "RESTORE FAILED: no FreeFem++ under $FFHOME; archive's first entries:"
  tar tzf "$FFROOT" 2>/dev/null | head -12 | sed 's/^/      /'
  return 1
}

# ------------------------------------------------------------------ p1: measured R2-7 ratio
do_p1() {
  require_preflight || { seg_end REFUSED "yes" "a passing preflight for this pin"; return 1; }
  local lvl d rc rows line
  say "P1 start instance=$INSTANCE (R2-7 constraints 2+3: the ratio must be MEASURED, never old-reading divided by 1000)"
  require_solver || return 1
  fetch model/scripts/gen_ns_re_edp.py || return 1
  fetch model/scripts/finalize_ns_truth.py || return 1
  fetch model/scripts/check_ns_re_to_stokes.py || return 1
  fetch model/cases/contraction_2d/cfd/C-base/C-base_stokes.edp || return 1
  fetch model/cases/contraction_2d/cfd/C-base/C-base_raw.csv || return 1
  say "P1 re-emitting the four NS .edp from the unit-fixed generator"
  if ! ( cd "$WS/model/scripts" && python3 gen_ns_re_edp.py --write ) >"$OUTD/p1_emit.txt" 2>&1; then
    say "P1 REFUSE: the generator itself refused to emit -- its diff/lint output is the finding"
    tail -6 "$OUTD/p1_emit.txt" | sed 's/^/      /'
    return 1
  fi
  grep -E "re-divisions=|diff proof OK" "$OUTD/p1_emit.txt" | tail -3 | sed 's/^/      /'
  for lvl in $NS_LEVELS; do
    d="$WS/model/cases/contraction_2d/cfd/C-base_ns_re$lvl"
    rc=0
    ( cd "$d" && timeout 90 "$FFBIN" -nw "C-base_ns_re$lvl.edp" ) >"$OUTD/p1_re$lvl.txt" 2>&1 || rc=$?
    rows=$(wc -l < "$d/C-base_ns_re${lvl}_raw.csv" 2>/dev/null || echo 0)
    say "P1 solve Re=$lvl rc=$rc raw_rows=$rows last_it=$(grep -c 'NS it=' "$OUTD/p1_re$lvl.txt")"
  done
  local unit_lines=0 fails=0
  for lvl in $NS_LEVELS; do
    rc=0
    ( cd "$WS/model/scripts" && python3 finalize_ns_truth.py --level "$lvl" ) >"$OUTD/p1_merge_$lvl.txt" 2>&1 || rc=$?
    line=$(grep -m1 -E '^\[unit\]|^\[FAIL\]|AWAITING' "$OUTD/p1_merge_$lvl.txt" | cut -c1-170)
    [ -n "${line:-}" ] && case "$line" in *"[unit]"*) unit_lines=$((unit_lines + 1));; esac
    [ "$rc" != 0 ] && fails=$((fails + 1))
    say "P1 merge Re=$lvl rc=$rc :: ${line:-NO [unit] LINE -- report missing, never as pass}"
  done
  say "P1 unit-lines=$unit_lines/$NS_COUNT merge_rc_nonzero=$fails  (constraint 2's reading is the Re=1e-3 line; nothing else substitutes for it)"
  # A segment that solved nothing must not leave the box looking like a success: this is the same
  # family as the E5 run that logged rc=127 as wall=1 ms, and it was reproduced locally just now
  # (p1 exited 0 with four AWAITING lines because the loops had nowhere to fail).
  if [ "$unit_lines" -eq 0 ]; then
    say "P1 REFUSE TO CONCLUDE: no [unit] line at any level -> the trip produced NO measured ratio; report it as missing"
    seg_end INCOMPLETE "yes, nothing measurable landed" "all four levels (solver did not run or files missing)"
    return 1
  fi
  if [ "$unit_lines" -lt "$NS_COUNT" ]; then
    say "P1 PARTIAL: $unit_lines/$NS_COUNT levels produced a ratio -- the rest stay unquoted, not back-filled"
    seg_end PARTIAL "yes" "$(( NS_COUNT - unit_lines )) levels without a [unit] line"
    return 0
  fi
  seg_end DONE "yes if no other phase is queued" "nothing in p1"
}

# ------------------------------------------------------------------ p2: K0b staged emission
do_p2() {
  require_preflight || { seg_end REFUSED "yes" "a passing preflight for this pin"; return 1; }
  local ph rc
  say "P2 start instance=$INSTANCE (K0b: syntax probe -> 12-digit staged emission -> reference scan)"
  require_solver || return 1
  fetch model/scripts/ops/run_k0b_5236655.sh || return 1
  fetch model/scripts/route2/generate_t_case.py || return 1
  fetch model/scripts/route2/k0_truth_gate.py || return 1
  fetch model/scripts/route2/artifacts.py || return 1
  fetch model/scripts/route2/residual_scorers.py || return 1
  fetch model/scripts/route2/t_geometry.py || return 1
  fetch model/scripts/route2/selftest_route2_stdlib.py || return 1
  for ph in check smoke run; do
    rc=0
    # fetch() lands files under $WS/<path>; $OUTD is only the log directory.  Calling the log
    # path gave rc=127 on the instance (14:41) -- loud, but a phase that never started.
    BUDGET_S=$(( SEGMENT_S / 4 )) bash "$WS/model/scripts/ops/run_k0b_5236655.sh" "$ph" >"$OUTD/p2_$ph.log" 2>&1 || rc=$?
    say "P2 k0b-$ph rc=$rc :: $(grep -m1 -E 'PROBE OK|no PROBE OK|companions written|ABORT|scan json|CONTROL|total=' "$OUTD/p2_$ph.log" | cut -c1-140)"
    if [ "$rc" != 0 ]; then
      say "P2 STOP after $ph -- the refusal IS the finding. If it is 'no PROBE OK' then floor is not executable on v4.9, so the staged reference is NOT_EMITTABLE and K0b closes as R3; that is a different conclusion from 'the cell is unmeasurable by nature' and must be reported as the former."
      grep -m3 -E 'ABORT|E\|' "$OUTD/p2_$ph.log" | cut -c1-150 | sed 's/^/      /'
      seg_end STOPPED "yes" "phases after k0b-$ph"
      return 1
    fi
  done
  seg_end DONE "yes" "nothing in p2"
}

# ------------------------------------------------------------------ p3: E5 unit cost, three guards
do_p3() {
  require_preflight || { seg_end REFUSED "yes" "a passing preflight for this pin"; return 1; }
  local i s e w rc ok=0 nrows
  say "P3 start instance=$INSTANCE -- E5 CFD unit cost: a thesis-side cell measured on route-2 machine time"
  require_solver || return 1
  fetch model/scripts/gen_ns_re_edp.py || return 1
  fetch model/cases/contraction_2d/cfd/C-base/C-base_stokes.edp || return 1
  fetch model/cases/contraction_2d/cfd/C-base/C-base_raw.csv || return 1
  # guard 0: the timed case is the SAME shipped Stokes solve with only its absolute output paths
  # redirected, by the already-tested helper.  Writing a small bespoke .edp would time a different
  # machine, and unverified FreeFEM syntax would fail for reasons that are not the solver's.
  if ! ( cd "$WS/model/scripts" && python3 -c "import pathlib,gen_ns_re_edp as gz; pathlib.Path('$OUTD/e5_stokes.edp').write_text(gz.probe_base((gz.CASE_DIR/'C-base_stokes.edp').read_text(encoding='utf-8').replace('\r\n','\n')),encoding='utf-8'); print('derived')" ) >"$OUTD/p3_derive.txt" 2>&1; then
    say "P3 REFUSE: could not derive the timed .edp"
    tail -4 "$OUTD/p3_derive.txt" | sed 's/^/      /'
    return 1
  fi
  say "P3 guard3 first: one UNTIMED solve of that exact file must return rc=0 AND write the 2113-row truth"
  rc=0
  ( cd "$OUTD" && timeout 300 "$FFBIN" -nw e5_stokes.edp ) >"$OUTD/e5_probe.txt" 2>&1 || rc=$?
  nrows=$(wc -l < "$OUTD/probe_syntax_raw.csv" 2>/dev/null || echo 0)
  say "P3 probe rc=$rc rows_written=$nrows"
  if [ "$rc" != 0 ] || [ "$nrows" -lt 2000 ]; then
    say "P3 REFUSE TO MEASURE: the probe did not produce a real solve, so no cost number is written this segment"
    seg_end REFUSED "yes" "a working solver run (see e5_probe.txt)"
    return 1
  fi
  : > "$OUTD/e5_runs.tsv"
  for i in $(seq 1 "$E5_ATTEMPTS"); do
    s=$(date +%s%N)
    rc=0
    ( cd "$OUTD" && timeout 300 "$FFBIN" -nw e5_stokes.edp ) >"$OUTD/e5_run$i.txt" 2>&1 || rc=$?
    e=$(date +%s%N); w=$(( (e - s) / 1000000 ))
    printf '%s\t%s\t%s\n' "$i" "$rc" "$w" >> "$OUTD/e5_runs.tsv"
    say "P3 run $i rc=$rc wall_ms=$w"
    [ "$rc" = 0 ] && ok=$((ok + 1))
  done
  say "P3 guard1: $ok/$E5_ATTEMPTS runs returned rc=0 -- failures are EXCLUDED from every number"
  if [ "$ok" -lt "$E5_MIN_OK" ]; then
    say "P3 REFUSE TO LAND: only $ok successful solves, the pre-registered rule needs >= $E5_MIN_OK -- a median over a sick machine is not a unit cost"
    seg_end REFUSED "yes" ">= $E5_MIN_OK successful solves"
    return 1
  fi
  python3 - "$OUTD/e5_runs.tsv" "$INSTANCE" "$FULL_PIN" <<'PY' || { say "P3 REFUSE TO LAND: summary writer failed"; return 1; }
import csv, json, hashlib, statistics, sys
from pathlib import Path
path, instance, pin = sys.argv[1:4]
rows = list(csv.reader(open(path), delimiter="\t"))
ok = [int(r[2]) for r in rows if r[1] == "0"]
out = {
    "what": "E5 same-machine CFD unit cost: wall time of one shipped Stokes solve",
    "provenance": "measured during route-2 machine time; the cell itself belongs to the thesis side",
    "case": "model/cases/contraction_2d/cfd/C-base/C-base_stokes.edp with absolute output paths redirected by gen_ns_re_edp.probe_base",
    "instance": instance, "pin": pin,
    "attempts": len(rows), "rc_nonzero_excluded": len(rows) - len(ok), "successes_used": len(ok),
    "landing_rule": ">= 5 of 7 rc=0 runs, else refuse to land",
    "wall_ms_median": statistics.median(ok), "wall_ms_min": min(ok), "wall_ms_max": max(ok),
    "wall_ms_all": ok,
    "statistic": "median, because run duration is right-skewed; min/max printed so the spread stays visible",
    "guard3_solver_probe": "one untimed rc=0 solve producing the 2113-row truth before any timing was kept",
}
dst = Path(path + ".summary.json")
dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("P3 LANDED %s %d B sha256=%s" % (dst, dst.stat().st_size,
                                       hashlib.sha256(dst.read_bytes()).hexdigest()[:16]))
print("P3 median_ms=%s min=%s max=%s n_used=%s excluded_rc_nonzero=%s"
      % (out["wall_ms_median"], out["wall_ms_min"], out["wall_ms_max"],
         out["successes_used"], out["rc_nonzero_excluded"]))
PY
  seg_end DONE "yes -- E5 landed; log it as route-2 time serving a thesis-side cell" "nothing in p3"
}

# ------------------------------------------------------------------ dispatch
TARGET="$MODE"
case "$MODE" in
  preflight) mkdir -p "$OUTD" 2>/dev/null; do_preflight; exit $? ;;
  restore)   mkdir -p "$OUTD" 2>/dev/null; do_restore; exit $? ;;
  p1|p2|p3)
    mkdir -p "$OUTD" 2>/dev/null || { say "FATAL: cannot create $OUTD"; exit 4; }
    say "launching $MODE as a child under timeout ${SEGMENT_S}s; progress in $OUTD/${MODE}.log"
    rc=0
    WS="$WS" OUTD="$OUTD" FULL_PIN="$FULL_PIN" FFROOT="$FFROOT" SEGMENT_S="$SEGMENT_S" \
    NS_COUNT="$NS_COUNT" FFHOME="$FFHOME" NS_LEVELS="$NS_LEVELS" E5_ATTEMPTS="$E5_ATTEMPTS" E5_MIN_OK="$E5_MIN_OK" TARGET="$MODE" \
      timeout "$SEGMENT_S" bash "$0" "${MODE}_body" >"$OUTD/${MODE}.log" 2>&1 || rc=$?
    tail -26 "$OUTD/${MODE}.log" | sed 's/^/    > /'
    [ "$rc" = 124 ] && say "$MODE was cut by the segment cap (${SEGMENT_S}s) -- report it as PARTIAL, not as success or failure"
    say "SEGMENT-END phase=$MODE child_rc=$rc wall_used=$(used)s segment_cap=${SEGMENT_S}s instance=$INSTANCE log=$OUTD/${MODE}.log"
    say "SEGMENT-END shutdown=$( [ "$rc" = 0 ] && echo 'yes if no other phase is queued' || echo 'yes, the segment is over' )"
    say "SEGMENT-END missing=$( [ "$rc" = 0 ] && echo none || echo 'see the lines above' )"
    exit "$rc" ;;
  p1_body) do_p1; exit $? ;;
  p2_body) do_p2; exit $? ;;
  p3_body) do_p3; exit $? ;;
  status)
    say "STATUS instance=$INSTANCE pin=${FULL_PIN:0:7} this-invocation-used=$(used)s out=$OUTD"
    ls -1 "$OUTD" 2>/dev/null | head -30
    grep -h -E "SEGMENT-END|P1 merge Re=|P3 LANDED|P2 k0b-" "$OUTD"/*.log 2>/dev/null | tail -14
    exit 0 ;;
  *) echo "usage: $0 {preflight|restore|p1|p2|p3|status}"; exit 2 ;;
esac
