# Shared shell for the measured optimisation cases.  Source it; do not run it.
#
# Every case runs its driver twice under the retained analysis pipeline
# (tools/drperf-dev run --blocks, then derive): once against a BASE tree and
# once against a FIX tree built by prepare.sh under $WORK, prints the cost
# formulas of the regions the case is about side by side, and then runs the
# case's equivalence check natively on both trees.

DRPERF=${DRPERF:-/home/ubuntu/drperf}
CASES_ROOT=${CASES_ROOT:-/home/ubuntu/drperf-cases}      # source trees + venvs of the original study
MC=$DRPERF/examples/measured_cases
WORK=${WORK:-$MC/work}                                  # per-case base/fix trees (prepare.sh)
OUT=${OUT_ROOT:-$MC/out}                                # drperf runs, derive output, dumps
DEV=$DRPERF/tools/drperf-dev
VLLM_PY=${VLLM_PY:-$DRPERF/third_party/vllm-cpu/.venv/bin/python}
WAN_PY=${WAN_PY:-$CASES_ROOT/videogen/.venv/bin/python}
PERFMARK_PATH=$DRPERF/perfmark/python:$DRPERF/build
THREADS=${THREADS:-8}
REPEAT=${REPEAT:-1}
MAX_SLOTS=${MAX_SLOTS:-2097152}

die() { echo "error: $*" >&2; exit 2; }

need_vllm() {
    [ -x "$VLLM_PY" ] || die "no vLLM venv at $VLLM_PY: run $DRPERF/third_party/vllm-cpu/setup.sh"
    [ -d "$DRPERF/third_party/hf/hub/models--facebook--opt-125m" ] || \
        die "facebook/opt-125m missing: HF_HOME=$DRPERF/third_party/hf python -c \"from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-125m')\""
    export VLLM_CPU_OMP_THREADS_BIND=all OMP_NUM_THREADS=$THREADS
}

need_wan() {
    [ -x "$WAN_PY" ] || die "no diffusers venv at $WAN_PY"
}

# dr_run NAME [extra drperf-dev run args] -- CMD...
# Runs CMD under drperf with per-block counting, then derives every region.
dr_run() {
    local name=$1; shift
    local extra=()
    while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
    mkdir -p "$OUT"
    rm -rf "$OUT/$name"
    echo "== run $name"
    local t0=$(date +%s)
    if ! "$DEV" run --blocks -q --threads "$THREADS" --repeat "$REPEAT" --max-slots "$MAX_SLOTS" \
            "${extra[@]}" -o "$OUT/$name" -- "$@" > "$OUT/$name.log" 2>&1; then
        tail -20 "$OUT/$name.log" >&2
        die "run $name failed (log: $OUT/$name.log)"
    fi
    echo "   ${name}: $(( $(date +%s) - t0 ))s instrumented (log $OUT/$name.log)"
    "$DEV" derive "$OUT/$name" > "$OUT/$name/derive.txt" 2>&1 || true
}

# formula NAME REGION -> the cost line(s) of REGION from NAME's derive output
formula() {
    awk -v r="$2" '$1=="derive" {cur=$2} cur==r && /^  cost\(/ {sub(/^  /, ""); print}' "$OUT/$1/derive.txt"
}

# compare BASE FIX REGION... -> before/after formulas per region (every regime)
compare() {
    local base=$1 fix=$2; shift 2
    echo "== formulas (instructions per call; blocks and irregular share in brackets)"
    for r in "$@"; do
        echo "  $r"
        formula "$base" "$r" | sed 's/  */ /g; s/^/    base: /'
        formula "$fix" "$r" | sed 's/  */ /g; s/^/    fix:  /'
    done
}

# percall BASE FIX REGION... -> mean instructions per call at every state point,
# from the run's per-trigger statistics (works with a single state point too)
percall() {
    local base=$1 fix=$2; shift 2
    echo "== mean instructions per call, by state point (self = own thread, nested regions excluded)"
    for r in "$@"; do
        echo "  $r"
        join -t'|' -j1 <(percall_table "$base" "$r") <(percall_table "$fix" "$r") 2>/dev/null | \
            awk -F'|' '{printf "    %-40s base %14s   fix %14s\n", $1, $2, $3}'
    done
}
percall_table() {   # NAME REGION [COL] -> "state|mean" lines, sorted; COL 8 = self, 6 = own thread incl. nested
    "$DEV" show "$OUT/$1" 2>/dev/null | awk -v r="$2" -v c="${3:-8}" '$1==r && $2 ~ /^@/ && $3 != "-" && $4 ~ /^[0-9]+$/ && $c ~ /^[0-9][0-9,]*[0-9]$|^[0-9]$/ {print $3 "|" $c}' | sort
}

# percall_incl BASE FIX REGION... -> the same, own thread INCLUDING nested regions
# (what a region costs its caller; the study's per-call numbers for parent regions)
percall_incl() {
    local base=$1 fix=$2; shift 2
    echo "== mean instructions per call, by state point (own thread, nested regions included)"
    for r in "$@"; do
        echo "  $r"
        join -t'|' -j1 <(percall_table "$base" "$r" 6) <(percall_table "$fix" "$r" 6) 2>/dev/null | \
            awk -F'|' '{printf "    %-40s base %14s   fix %14s\n", $1, $2, $3}'
    done
}

# percall_full BASE FIX REGION... -> total / own thread / other threads per call, per state point
percall_full() {
    local base=$1 fix=$2; shift 2
    echo "== mean instructions per call, by state point: total = own thread + other threads while open"
    for r in "$@"; do
        echo "  $r"
        for c in 5 6 7; do
            case $c in 5) l=total;; 6) l=own-thr;; 7) l=other-thr;; esac
            join -t'|' -j1 <(percall_table "$base" "$r" $c) <(percall_table "$fix" "$r" $c) 2>/dev/null | \
                awk -F'|' -v l="$l" '{printf "    %-36s %-9s base %14s   fix %14s\n", $1, l, $2, $3}'
        done
    done
}

# perrun BASE FIX REGION... -> total / own / other per call for each grid point,
# pairing the runs by position (for a fix that changes what a state means)
perrun() {
    local base=$1 fix=$2; shift 2
    echo "== mean instructions per call per grid point: total = own thread + other threads while open"
    for r in "$@"; do
        echo "  $r"
        local i=0
        for fb in "$OUT/$base"/run*_r0.*.json; do
            local ff; ff=$(ls "$OUT/$fix"/run$(printf %03d $i)_r0.*.json 2>/dev/null | head -1)
            [ -n "$ff" ] || break
            printf "    run%d  base %s\n           fix  %s\n" $i \
                "$("$DEV" show "$fb" 2>/dev/null | awk -v r="$r" '$1==r && $2 ~ /^@/ && $3 != "-" && $4 ~ /^[0-9]+$/ {printf "[%s] total %s  own %s  other %s", $3, $5, $6, $7}')" \
                "$("$DEV" show "$ff" 2>/dev/null | awk -v r="$r" '$1==r && $2 ~ /^@/ && $3 != "-" && $4 ~ /^[0-9]+$/ {printf "[%s] total %s  own %s  other %s", $3, $5, $6, $7}')"
            i=$((i+1))
        done
    done
}

# ---------------------------------------------------------------- trees

# tree_copy SRC DST [hard]  -- plain copy, or a hardlink farm for big trees
tree_copy() {
    [ -d "$1" ] || die "source tree $1 missing (set CASES_ROOT)"
    rm -rf "$2"; mkdir -p "$(dirname "$2")"
    if [ "${3:-}" = hard ]; then cp -al "$1" "$2"; else cp -a "$1" "$2"; fi
}

# unshare TREE FILE...  -- give hardlinked files their own inode before editing
unshare() {
    local tree=$1; shift
    for f in "$@"; do
        [ -f "$tree/$f" ] || continue
        cp -p "$tree/$f" "$tree/$f.tmp$$" && mv -f "$tree/$f.tmp$$" "$tree/$f"
    done
}

# patch_state TREE STRIP PATCH -> on | off | unknown
patch_state() {
    if (cd "$1" && patch -p"$2" -R --dry-run -s -f < "$3" > /dev/null 2>&1); then echo on
    elif (cd "$1" && patch -p"$2" --dry-run -s -f < "$3" > /dev/null 2>&1); then echo off
    else echo unknown; fi
}

# ensure_patch TREE STRIP PATCH on|off
ensure_patch() {
    local tree=$1 strip=$2 p=$3 want=$4
    local have; have=$(patch_state "$tree" "$strip" "$p")
    [ "$have" = unknown ] && die "$(basename "$p") neither applies nor reverses on $tree"
    [ "$have" = "$want" ] && return 0
    unshare "$tree" $(grep '^+++ ' "$p" | awk '{print $2}' | cut -d/ -f$((strip+1))-)
    if [ "$want" = on ]; then (cd "$tree" && patch -p"$strip" -s -f < "$p")
    else (cd "$tree" && patch -p"$strip" -R -s -f < "$p"); fi
    echo "   $(basename "$p"): $have -> $want on $(basename "$tree")"
}

# dump_equal A B -> compares two native output dumps
dump_equal() {
    if cmp -s "$1" "$2"; then echo "== equivalence: outputs identical ($(basename "$1") = $(basename "$2"))"
    else echo "== equivalence: OUTPUTS DIFFER: $1 $2"; return 1; fi
}
