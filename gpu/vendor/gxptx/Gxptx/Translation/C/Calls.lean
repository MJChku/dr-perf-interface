/-
  Gxptx.Translation.C.Calls — generated-C call-site and external-adapter
  renderers for PTX call parameter buffers.
-/
import Gxptx.Translation.C.Operands
namespace Gxptx

/- PTX definition: `call` transfers control to a direct or indirect function and passes arguments through declared parameter objects. -/
/- Lean modeling: the emitted pointer array preserves input order and the existing `void **` GX ABI; indirect calls only add the historical function-pointer cast, so generated text is unchanged. -/
def emitInputCall (callee : String) (indirect : Bool) (inputs : List String) : String :=
  let invoke (argv : String) :=
    if indirect then
      s!"((void(*)(void**))(uintptr_t){oC callee})({argv});"
    else
      s!"{callee}({argv});"
  if inputs.isEmpty then
    "    " ++ invoke "nullptr"
  else
    let ptrs := inputs.map (fun p => s!"(void*)(uintptr_t)({oC p})")
    "    { void* _gx_call_args[] = { " ++ String.intercalate ", " ptrs ++
      " }; " ++ invoke "_gx_call_args" ++ " }"

/- PTX definition: PTX modules may declare externally defined functions and invoke them with `call`. -/
/- Lean modeling: the closed table routes only adapters whose PTX parameter-buffer ABI is implemented; returning `none` for unknown externals prevents an unsound host-ABI guess. -/
def externalCallTarget : String → Option String
  | "vprintf"      => some "gx_ptx_vprintf"
  | "__assertfail" => some "gx_ptx_assertfail"
  | _              => none

/- PTX definition: return parameters and input parameters of `call` are distinct ordered parameter tuples. -/
/- Lean modeling: the two arrays preserve PTX tuple order and absence as `nullptr`; adapter behavior and compilation remain explicit trusted runtime boundaries. -/
def emitExternalCall (target : String) (outputs inputs : List String) : String :=
  let arrayDecl (name : String) (params : List String) :=
    if params.isEmpty then ("", "nullptr")
    else
      let ptrs := params.map (fun p => s!"(void*)(uintptr_t)({oC p})")
      (s!"void* {name}[] = \{" ++ String.intercalate ", " ptrs ++ " }; ", name)
  let (retDecl, retArg) := arrayDecl "_gx_call_rets" outputs
  let (argDecl, argArg) := arrayDecl "_gx_call_args" inputs
  "    { " ++ retDecl ++ argDecl ++ s!"{target}({retArg}, {argArg});" ++ " }"

/- GX extension: one reduced payload pointer, including its exact dynamic
   global-memory instruction count. The generic gx_ptx_memory_event marker
   shares this runtime ABI without opting the module into NCCL's direct-control
   load/store lowering. -/
def emitPayloadEventCall (inputs : List String) : String :=
  match inputs with
  | address :: bytes :: isLoad :: instructions :: _ =>
      s!"    gx_nccl_payload_event_v2(*(const uint64_t*)(uintptr_t)({oC address}), *(const uint64_t*)(uintptr_t)({oC bytes}), *(const uint32_t*)(uintptr_t)({oC isLoad}), *(const uint64_t*)(uintptr_t)({oC instructions}));"
  | _ => "    gx_ptx_trap_op(\"gx_ptx_payload_event.abi\");"

/- GX extension: delimit exact reduceCopy accesses for count validation. -/
def emitPayloadScopeCall (inputs : List String) : String :=
  match inputs with
  | enabled :: _ =>
      s!"    gx_nccl_payload_scope(*(const uint32_t*)(uintptr_t)({oC enabled}));"
  | _ => "    gx_ptx_trap_op(\"gx_ptx_payload_scope.abi\");"

/- GX annotation: the guarded region computes only payload values for this
   destination, never control or communication progress. Native CUDA returns
   one; GX may omit these values only when the destination has no backing. -/
def emitPayloadNeededCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [result], [address] =>
      s!"    *(uint32_t*)(uintptr_t)({oC result}) = !memory_mg_is_fake((void*)(uintptr_t)(*(const uint64_t*)(uintptr_t)({oC address})));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_payload_needed.abi\");"

/- GX annotation: exact block scan, with lane prefix and block total packed in
   the low/high words. Inputs and output use PTX parameter buffers. -/
def emitBlockScanCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [result], [value, exclusive] =>
      s!"    *(uint64_t*)(uintptr_t)({oC result}) = gx_block_scan_u32(*(const uint32_t*)(uintptr_t)({oC value}), *(const uint32_t*)(uintptr_t)({oC exclusive}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_block_scan_u32.abi\");"

/- GX annotation: an explicitly identified control poll is a real atomic load,
   not payload bandwidth. Unannotated loads keep their ordinary accounting. -/
def emitPollLoadCall (wide : Bool) (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [result], [address] =>
      let ty := if wide then "uint64_t" else "uint32_t"
      s!"    *({ty}*)(uintptr_t)({oC result}) = gx_ptx_poll_load<{ty}>(*(const uint64_t*)(uintptr_t)({oC address}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_poll_load.abi\");"

/- Semantic cooperative-copy marker: parameter widths follow its native
   uint64_t/uint32_t ABI. Only explicitly annotated tile copies use this path. -/
def emitBlockCopy16Call (inputs : List String) : String :=
  match inputs with
  | [dst, src, count, lane, lanes, direction] =>
      let wide := [dst, src, count].map (fun p => s!"*(const uint64_t*)(uintptr_t)({oC p})")
      let narrow := [lane, lanes, direction].map (fun p => s!"*(const uint32_t*)(uintptr_t)({oC p})")
      "    gx_ptx_block_copy16(" ++ String.intercalate ", " (wide ++ narrow) ++ ");"
  | _ => "    gx_ptx_trap_op(\"gx_ptx_block_copy16.abi\");"

def emitUniformLoad64Call (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [result], [address, lane, lanes] =>
      s!"    *(uint64_t*)(uintptr_t)({oC result}) = gx_ptx_uniform_load64(*(const uint64_t*)(uintptr_t)({oC address}), *(const uint32_t*)(uintptr_t)({oC lane}), *(const uint32_t*)(uintptr_t)({oC lanes}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_uniform_load64.abi\");"

def emitCountedLoad32Call (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [result], [address, count] =>
      s!"    *(uint32_t*)(uintptr_t)({oC result}) = gx_ptx_counted_load32(*(const uint64_t*)(uintptr_t)({oC address}), *(const uint32_t*)(uintptr_t)({oC count}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_counted_load32.abi\");"

def emitCountedLoad8Call (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [result], [address, count] =>
      s!"    *(uint32_t*)(uintptr_t)({oC result}) = gx_global_load_u8_repeated(*(const uint64_t*)(uintptr_t)({oC address}), *(const uint32_t*)(uintptr_t)({oC count}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_counted_load8.abi\");"

def emitCohortBarrierCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [], [participants] =>
      s!"    barrier_sync(0, *(const uint32_t*)(uintptr_t)({oC participants}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_cohort_barrier.abi\");"

/- Explicit annotation for a one-dimensional block of cooperative operations.
   Native CUDA uses individual lanes; GX may use a single representative only
   where the caller preserves all data, access multiplicity and control. -/
def emitBlockCohortWidthCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [result], [] =>
      s!"    *(uint32_t*)(uintptr_t)({oC result}) = gx_dev_blockDim().x;"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_block_cohort_width.abi\");"

def emitWarpCohortWidthCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [result], [] => s!"    *(uint32_t*)(uintptr_t)({oC result}) = 32u;"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_warp_cohort_width.abi\");"

/- GX annotation: scan a shared-memory matrix column-wise, retain prefixes
   only at entries equal to one, and store one real global total per column.
   Surrounding barriers belong to the caller, not to this adapter. -/
def emitMaskedColumnScanCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [], [tile, totals, columns] =>
      s!"    gx_ptx_masked_column_scan_u32(*(const uint64_t*)(uintptr_t)({oC tile}), *(const uint64_t*)(uintptr_t)({oC totals}), *(const uint32_t*)(uintptr_t)({oC columns}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_masked_column_scan_u32.abi\");"

def emitBlockTransferU32Call (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [], [dst, src, count, extent, mode] =>
      s!"    gx_ptx_block_transfer_u32(*(const uint64_t*)(uintptr_t)({oC dst}), *(const uint64_t*)(uintptr_t)({oC src}), *(const uint32_t*)(uintptr_t)({oC count}), *(const uint32_t*)(uintptr_t)({oC extent}), *(const uint32_t*)(uintptr_t)({oC mode}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_block_transfer_u32.abi\");"

def emitMaskedColumnAddCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [], [matrix, bias1, bias2, shared, overflow, count, columns, limit] =>
      s!"    gx_ptx_masked_column_add_u32(*(const uint64_t*)(uintptr_t)({oC matrix}), *(const uint64_t*)(uintptr_t)({oC bias1}), *(const uint64_t*)(uintptr_t)({oC bias2}), *(const uint64_t*)(uintptr_t)({oC shared}), *(const uint64_t*)(uintptr_t)({oC overflow}), *(const uint32_t*)(uintptr_t)({oC count}), *(const uint32_t*)(uintptr_t)({oC columns}), *(const uint64_t*)(uintptr_t)({oC limit}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_masked_column_add_u32.abi\");"

def emitCountedCopyCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [], [dst, src, bytes, width, isLoad] =>
      s!"    gx_ptx_counted_copy(*(const uint64_t*)(uintptr_t)({oC dst}), *(const uint64_t*)(uintptr_t)({oC src}), *(const uint64_t*)(uintptr_t)({oC bytes}), *(const uint32_t*)(uintptr_t)({oC width}), *(const uint32_t*)(uintptr_t)({oC isLoad}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_counted_copy.abi\");"

def emitSharedIndexGatherCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [], [dst, src, indices, first, last, count, lane, stride] =>
      s!"    gx_ptx_shared_index_gather_f32(*(const uint64_t*)(uintptr_t)({oC dst}), *(const uint64_t*)(uintptr_t)({oC src}), *(const uint64_t*)(uintptr_t)({oC indices}), *(const uint32_t*)(uintptr_t)({oC first}), *(const uint32_t*)(uintptr_t)({oC last}), *(const uint32_t*)(uintptr_t)({oC count}), *(const uint32_t*)(uintptr_t)({oC lane}), *(const uint32_t*)(uintptr_t)({oC stride}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_shared_index_gather_f32.abi\");"

/- Independent predicate columns have no intervening memory side effects.
   Native CUDA retains its ballots; GX shares their fiber rendezvous. -/
def emitWarpBallotColumnsCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [], [predicates, columns, results] =>
      s!"    gx_warp_ballot_columns(*(const uint32_t*)(uintptr_t)({oC predicates}), *(const uint32_t*)(uintptr_t)({oC columns}), (uint32_t*)(uintptr_t)(*(const uint64_t*)(uintptr_t)({oC results})));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_ballot_columns.abi\");"

def emitMbarrierWaitCall (outputs inputs : List String) : String :=
  match outputs, inputs with
  | [], [address, token, parity] =>
      s!"    gx_mbarrier_wait(*(const uint64_t*)(uintptr_t)({oC address}), *(const uint64_t*)(uintptr_t)({oC token}), *(const uint32_t*)(uintptr_t)({oC parity}));"
  | _, _ => "    gx_ptx_trap_op(\"gx_ptx_mbarrier_wait.abi\");"

end Gxptx
