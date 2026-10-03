#!/bin/bash
# prepare.sh [CASE...]  -- build the BASE and FIX source trees of each case
# under $WORK from the marked trees of the original study ($CASES_ROOT).
#
# A case's FIX tree carries its fix; its BASE tree is the same marked tree
# with the fix taken out (or, for the Wan trees, re-marked from the pristine
# state).  Nothing under $CASES_ROOT is modified: vLLM trees are hardlink farms
# in which only the patched files are unshared; Wan trees are plain copies.
set -euo pipefail
. "$(dirname "$0")/lib.sh"

ALL="vllm_admission vllm_prepare_inputs vllm_input_batch vllm_output_path vllm_logprobs wan_host wan_more wan_xattn wan_t5"
[ $# -gt 0 ] || set -- $ALL

# vllm_pair CASE SRC_TREE PATCH...  -- base: every patch off; fix: every patch on
vllm_pair() {
    local case=$1 src=$CASES_ROOT/$2; shift 2
    for v in base fix; do
        local t=$WORK/$case-$v
        tree_copy "$src" "$t" hard
        for p in "$@"; do ensure_patch "$t" 1 "$MC/$case/$p" $([ $v = fix ] && echo on || echo off); done
    done
}

# wan_tree CASE VARIANT SRC  -- plain copy of a marked diffusers tree
wan_tree() { tree_copy "$CASES_ROOT/videogen/$3" "$WORK/$1-$2"; }

# region_names TREE -> sorted region names declared in the tree
region_names() { grep -rhoE 'perfmark\.region\("[^"]+"' "$1" --include='*.py' | sort -u; }

for case in "$@"; do
    echo "== prepare $case"
    case $case in
    vllm_admission)      vllm_pair $case vllm-cpu-req   admission.patch ;;
    vllm_prepare_inputs) vllm_pair $case vllm-cpu-rb    prepare_inputs.patch cached_request_data.patch ;;
    vllm_input_batch)    vllm_pair $case vllm-cpu-batch input_batch.patch ;;
    vllm_output_path)
        # the study's tree carries the fix in its unmarked snapshot (.out.orig):
        # unmark, reverse the patch for the base, re-mark both with the clean two-region set
        for v in base fix; do
            t=$WORK/$case-$v
            tree_copy "$CASES_ROOT/vllm-cpu-out" "$t" hard
            unshare "$t" vllm/v1/engine/output_processor.py vllm/v1/engine/detokenizer.py vllm/entrypoints/offline_utils.py \
                         vllm/v1/engine/output_processor.py.out.orig vllm/v1/engine/detokenizer.py.out.orig vllm/entrypoints/offline_utils.py.out.orig
            "$VLLM_PY" "$MC/$case/mark_out_clean.py" "$t" --undo > /dev/null
            ensure_patch "$t" 1 "$MC/$case/output_path.patch" $([ $v = fix ] && echo on || echo off)
            # the marker restores every file from its .out.orig snapshot before
            # marking, so the snapshot has to carry the base/fix state too
            cp -p "$t/vllm/v1/engine/output_processor.py" "$t/vllm/v1/engine/output_processor.py.out.orig"
            "$VLLM_PY" "$MC/$case/mark_out_clean.py" "$t" > /dev/null
            # the study's third beat was measured with process_outputs alone
            "$VLLM_PY" "$MC/$case/unwrap_region.py" "$t/vllm/v1/engine/output_processor.py" make_request_output > /dev/null
            echo "   fix lines in output_processor.py ($v): $(grep -c 'iteration_stats is not None' "$t/vllm/v1/engine/output_processor.py") (base 1 = the pre-existing assert, fix 2)"
        done ;;
    vllm_logprobs)
        # the fix is applied by a script (indentation-agnostic), not a patch
        for v in base fix; do
            t=$WORK/$case-$v
            tree_copy "$CASES_ROOT/vllm-cpu-spec" "$t" hard
            unshare "$t" vllm/v1/sample/metadata.py vllm/v1/worker/gpu_input_batch.py vllm/v1/sample/sampler.py
            if ls "$t"/vllm/v1/sample/*.lp.orig > /dev/null 2>&1; then "$VLLM_PY" "$MC/$case/apply_logprobs_rows.py" "$t" --undo > /dev/null; fi
            [ $v = fix ] && "$VLLM_PY" "$MC/$case/apply_logprobs_rows.py" "$t" > /dev/null
            # the study's `stop` marker set (gather_logprobs, check_stop, ...) on top of the base regions
            unshare "$t" vllm/v1/engine/detokenizer.py vllm/v1/engine/logprobs.py vllm/v1/sample/logits_processor/builtin.py \
                         vllm/v1/engine/detokenizer.py.stop.orig vllm/v1/engine/logprobs.py.stop.orig vllm/v1/sample/logits_processor/builtin.py.stop.orig vllm/v1/sample/sampler.py.stop.orig
            "$VLLM_PY" "$MC/$case/mark_spec.py" "$t" stop > /dev/null
            echo "   logprobs rows fix: $v; gather_logprobs regions: $(grep -c 'perfmark.region("gather_logprobs"' "$t/vllm/v1/sample/sampler.py")"
        done ;;
    wan_host)
        # wan-src = pristine diffusers + the four host fixes, then marked.
        # base: unmark, take the fixes out at the unmarked level, re-mark.
        wan_tree $case fix wan-src
        wan_tree $case base wan-src
        t=$WORK/$case-base
        "$WAN_PY" "$MC/$case/mark_wan.py" "$t/diffusers" --undo > /dev/null
        for p in wan_rope wan_text_embed wan_vae_cat wan_misc; do ensure_patch "$t" 2 "$MC/$case/$p.patch" off; done
        "$WAN_PY" "$MC/$case/mark_wan.py" "$t/diffusers" > /dev/null ;;
    wan_more)
        # fix = wan-more (round-one fixes + round-two fixes, 44 regions);
        # base = wan-src (round-one fixes, 20 regions) + the 24 round-two regions
        wan_tree $case fix wan-more
        wan_tree $case base wan-src
        "$WAN_PY" "$MC/$case/mark_more.py" "$WORK/$case-base/diffusers" > /dev/null
        if ! diff <(region_names "$WORK/$case-base/diffusers") <(region_names "$WORK/$case-fix/diffusers") > /dev/null; then
            echo "   WARNING: region sets differ between base and fix:"; diff <(region_names "$WORK/$case-base/diffusers") <(region_names "$WORK/$case-fix/diffusers") | sed 's/^/     /' || true
        fi ;;
    wan_xattn)
        # wan_tax.patch was made against the marked tree (-p1) and is applied in wan-tax
        wan_tree $case fix wan-tax
        wan_tree $case base wan-tax
        ensure_patch "$WORK/$case-base" 1 "$MC/$case/wan_tax.patch" off ;;
    wan_t5)
        # mark_t5.py rewrites _get_t5_prompt_embeds from a template: without
        # --fix the original padding, with --fix the short-pad encode
        wan_tree $case base wan-t5
        wan_tree $case fix wan-t5
        "$WAN_PY" "$MC/$case/mark_t5.py" "$WORK/$case-base/diffusers" sq2 > /dev/null
        "$WAN_PY" "$MC/$case/mark_t5.py" "$WORK/$case-fix/diffusers" sq2 --fix > /dev/null ;;
    *) die "unknown case $case (one of: $ALL)" ;;
    esac
    echo "   $WORK/$case-base  $WORK/$case-fix"
done
