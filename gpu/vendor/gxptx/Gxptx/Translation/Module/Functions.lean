/-
  Gxptx.Translation.Module.Functions — byte-level PTX function extraction,
  signature decoding, argument aliases, and C function-body assembly.
-/
import Gxptx.Translation.Module.Registers
import Gxptx.Translation.Module.Shared
import Std.Data.HashSet
set_option linter.deprecated false
namespace Gxptx

/- PTX definition: PTX function definitions delimit their instruction bodies with braces. -/
/- Lean modeling: the literal ASCII value 123 enables allocation-free byte scanning and does not reinterpret source content. -/
private def bLBrace : UInt8 := 123  -- '{'
/- PTX definition: PTX function definitions delimit their instruction bodies with braces. -/
/- Lean modeling: the literal ASCII value 125 pairs with `bLBrace` for exact nesting and does not alter emitted text. -/
private def bRBrace : UInt8 := 125  -- '}'
/- PTX definition: kernel and device-function parameters are declared in parenthesized signatures. -/
/- Lean modeling: the literal ASCII value 40 enables exact byte-level signature scanning without introducing a separate PTX AST. -/
private def bLParen : UInt8 := 40   -- '('
/- PTX definition: kernel and device-function parameter declarations form a parenthesized signature. -/
/- Lean modeling: the literal ASCII value 41 pairs with `bLParen` so nested signature groups are sliced at their exact source boundary. -/
private def bRParen : UInt8 := 41   -- ')'

/- PTX definition: PTX source is a textual program whose tokens are recovered from contiguous source ranges. -/
/- Lean modeling: all call sites slice previously scanned token/body boundaries, so direct UTF-8 decoding preserves the original source bytes and avoids character-list allocation. -/
def bSlice (bs : ByteArray) (a b : Nat) : String :=
  String.fromUTF8! (bs.extract a b)

/- PTX definition: `.entry` and `.func` directive keywords introduce kernel/function declarations. -/
/- Lean modeling: bounded byte equality recognizes only the requested literal spelling; surrounding source-position validity remains the responsibility of the single-pass scanner. -/
def atKw (bs : ByteArray) (i : Nat) (kw : ByteArray) : Bool := Id.run do
  if i + kw.size > bs.size then return false
  for j in [0:kw.size] do
    if bs[i+j]! != kw[j]! then return false
  return true

/- PTX definition: a PTX function definition encloses its body in a balanced brace-delimited block. -/
/- Lean modeling: the depth counter preserves nested inline blocks and returns the source end on malformed input; full grammar validation remains trusted parsing. -/
partial def matchBrace (bs : ByteArray) (start : Nat) : Nat := Id.run do
  let mut depth := 0
  let mut i := start
  while i < bs.size do
    let c := bs[i]!
    if c == bLBrace then depth := depth + 1
    else if c == bRBrace then
      depth := depth - 1
      if depth == 0 then return i + 1
    i := i + 1
  return bs.size

/- PTX definition: parameter and return-parameter declaration lists are parenthesized. -/
/- Lean modeling: the depth counter handles nested declaration syntax and returns the source end on malformed input; it performs shape recovery rather than semantic validation. -/
partial def matchParen (bs : ByteArray) (start : Nat) : Nat := Id.run do
  let mut depth := 0
  let mut i := start
  while i < bs.size do
    let c := bs[i]!
    if c == bLParen then depth := depth + 1
    else if c == bRParen then
      depth := depth - 1
      if depth == 0 then return i + 1
    i := i + 1
  return bs.size

/- PTX definition: directive/function syntax uses fixed delimiters such as `(`, `{`, and `;`. -/
/- Lean modeling: a bounded linear scan preserves exact delimiter positions and returns the source end when absent so callers can reject/truncate malformed shapes safely. -/
def scanTo (bs : ByteArray) (i : Nat) (c : UInt8) : Nat := Id.run do
  let mut k := i
  while k < bs.size && bs[k]! != c do k := k + 1
  return k

/- PTX definition: spaces, tabs, and line breaks separate source tokens without becoming identifier content. -/
/- Lean modeling: recognizing exactly ASCII space/tab/newline/carriage-return preserves token bytes and supports allocation-free directive scanning. -/
def skipWs (bs : ByteArray) (i : Nat) : Nat := Id.run do
  let mut k := i
  while k < bs.size && (let c := bs[k]!; c == 32 || c == 9 || c == 10 || c == 13) do
    k := k + 1
  return k

/- PTX definition: `.entry` and `.func` introduce prototypes or definitions with signatures and optional bodies. -/
/- Lean modeling: one byte pass preserves source slices, skips semicolon-terminated prototypes, and keeps the first repeated definition; completeness and malformed-input handling remain trusted parsing. -/
partial def extractFnsRaw (src : String) : Array (String × String × String × Bool) := Id.run do
  let bs := src.toUTF8
  let n := bs.size
  let kwEntry := ".entry".toUTF8
  let kwFunc  := ".func".toUTF8
  let mut res : Array (String × String × String × Bool) := #[]
  let mut seen : Std.HashSet String := {}
  let mut i := 0
  while i < n do
    let kernel := atKw bs i kwEntry
    let func   := atKw bs i kwFunc
    if kernel || func then
      let kwlen := if kernel then 6 else 5
      let mut nameStart := skipWs bs (i + kwlen)
      -- A PTX function may begin with a parenthesized return-parameter list.
      if func && nameStart < n && bs[nameStart]! == bLParen then
        nameStart := skipWs bs (matchParen bs nameStart)
      let paren := scanTo bs nameStart bLParen
      let parenEnd := matchParen bs paren
      let nm := strim (bSlice bs nameStart paren)
      let bo := scanTo bs i bLBrace
      let semi := scanTo bs i 59
      -- Prototypes (`.extern .func ...;`) have no body.  Do not consume the
      -- next initializer or function body as if it belonged to the prototype.
      if semi < bo then
        i := semi + 1
      else if bo < n then
        let e := matchBrace bs bo
        -- concatenated TUs repeat functions; keep the first definition only
        if ! seen.contains nm then
          seen := seen.insert nm
          let params := if paren < n && parenEnd ≤ n then bSlice bs (paren+1) (parenEnd-1) else ""
          res := res.push (nm, params, bSlice bs (bo+1) (e-1), kernel)
        i := e
      else i := i + kwlen
    else i := i + 1
  return res

/- PTX definition: shared storage and function definitions belong to a specific versioned PTX module. -/
/- Lean modeling: translating each `ptxModules` segment independently prevents cross-image layout leakage and deduplicates concatenated definitions; fatbin extraction remains trusted. -/
def extractFns (src : String) : Array (String × String × String × Bool × Nat) := Id.run do
  let mut res : Array (String × String × String × Bool × Nat) := #[]
  let mut seen : Std.HashSet String := {}
  for image in ptxModules src do
    let total := sharedTotal image
    for (nm, sig, body, kernel) in extractFnsRaw image do
      -- nvcc retains definitions for the source markers, but calls lower
      -- directly to GX runtime accounting; do not emit their empty bodies.
      let payloadEvent := (nm.splitOn "gx_ptx_payload_event").length > 1
      let memoryEvent := (nm.splitOn "gx_ptx_memory_event").length > 1
      let gridSync := (nm.splitOn "gx_ptx_grid_sync").length > 1
      let blockScan := (nm.splitOn "gx_ptx_block_scan_u32").length > 1 ||
                       (nm.splitOn "gx_ptx_masked_column_scan_u32").length > 1
      let blockCopy := (nm.splitOn "gx_ptx_block_copy16").length > 1 ||
                       (nm.splitOn "gx_ptx_block_transfer_u32").length > 1 ||
                       (nm.splitOn "gx_ptx_masked_column_add_u32").length > 1 ||
                       (nm.splitOn "gx_ptx_counted_copy").length > 1 ||
                       (nm.splitOn "gx_ptx_shared_index_gather_f32").length > 1 ||
                       (nm.splitOn "gx_ptx_ballot_columns").length > 1 ||
                       (nm.splitOn "gx_ptx_mbarrier_wait").length > 1
      let uniformLoad := (nm.splitOn "gx_ptx_uniform_load64").length > 1 ||
                         (nm.splitOn "gx_ptx_counted_load32").length > 1 ||
                         (nm.splitOn "gx_ptx_counted_load8").length > 1 ||
                         (nm.splitOn "gx_ptx_cohort_barrier").length > 1 ||
                         (nm.splitOn "gx_ptx_block_cohort_width").length > 1 ||
                         (nm.splitOn "gx_ptx_warp_cohort_width").length > 1 ||
                         (nm.splitOn "gx_ptx_independent_block_cohort_width").length > 1
      let payloadScope := (nm.splitOn "gx_ptx_payload_scope").length > 1
      let abortPoll := (nm.splitOn "gx_ptx_nccl_abort_poll").length > 1
      let payloadNeeded := (nm.splitOn "gx_ptx_payload_needed").length > 1
      let pollLoad := ["gx_ptx_control_load_16", "gx_ptx_control_poll", "gx_ptx_control_load_acquire_sys_u32", "gx_ptx_control_load_sys_u32", "gx_ptx_poll_sys_u32", "gx_ptx_poll_sys_u64", "gx_ptx_poll_gpu_u64"].any
        (fun marker => (nm.splitOn marker).length > 1)
      if !payloadEvent && !memoryEvent && !gridSync && !blockScan && !blockCopy && !uniformLoad && !payloadScope && !abortPoll && !payloadNeeded && !pollLoad && ! seen.contains nm then
        seen := seen.insert nm
        res := res.push (nm, sig, body, kernel, total)
  return res

/- PTX definition: function signatures declare ordered parameter-space input objects. -/
/- Lean modeling: comma splitting followed by exact `.param` filtering preserves declaration order for the GX `void **args` ABI; richer grammar parsing remains trusted. -/
def signatureParamDecls (sig : String) : List String :=
  (sig.splitOn ",").filter (fun decl => ((strim decl).splitOn ".param").length > 1)

/- PTX definition: parameters have identifiers and aggregate by-value objects may use array declaration syntax. -/
/- Lean modeling: normalizing whitespace and taking the final declaration token preserves nvcc signatures; stripping only brackets/extent yields the C alias name used by loads. -/
def signatureParamName (decl : String) : String :=
  let toks := ((strim decl).replace "\n" " " |>.replace "\t" " ").splitOn " "
    |>.filter (! ·.isEmpty)
  let spec := toks.getLastD ""
  ((spec.replace "]" "").splitOn "[").headD spec

/- PTX definition: kernel and device-function parameters are addressable parameter-space objects. -/
/- Lean modeling: enumeration maps declaration order directly to GX `args[i]` slots and preserves source names, while the `void **` calling convention remains trusted ABI plumbing. -/
def signatureParamAliases (sig : String) : String :=
  String.join ((signatureParamDecls sig).enum.map (fun (i, decl) =>
    let nm := signatureParamName decl
    s!"    unsigned char* {nm} = (unsigned char*)args[{i}];\n"))

/- PTX definition: parameter storage size follows its fundamental type width and optional array extent. -/
/- Lean modeling: multiplying the decoded element bytes by the extent preserves Runtime-API argument sizes; unknown tokens retain the legacy 32-bit fallback and ABI packing remains trusted. -/
def signatureParamSize (decl : String) : Nat :=
  let d := strim decl
  let toks := (d.replace "\n" " " |>.replace "\t" " ").splitOn " " |>.filter (! ·.isEmpty)
  let ty := (toks.filterMap (fun raw =>
    let t := raw.replace "." ""
    if ["b8","u8","s8","b16","u16","s16","f16","bf16",
        "b32","u32","s32","f32","b64","u64","s64","f64"].contains t
    then some t else none)).getLastD "b32"
  let elem := if ty.endsWith "8" then 1 else if ty.endsWith "16" then 2
              else if ty.endsWith "64" then 8 else 4
  let spec := toks.getLastD ""
  match (spec.replace "]" "").splitOn "[" with
  | [_, n] => elem * n.toNat?.getD 0
  | _ => elem

/- PTX definition: kernel parameters form an ordered parameter-space declaration list. -/
/- Lean modeling: filtering `.param` declarations and applying `signatureParamSize` preserves the one-size-per-`args[i]` metadata order expected by GX launch plumbing. -/
def signatureParamSizes (sig : String) : List Nat :=
  (sig.splitOn ",").filterMap (fun decl =>
    let d := strim decl
    if (d.splitOn ".param").length > 1 then some (signatureParamSize d) else none)

/- PTX definition: packed launch parameters begin at their declared alignment,
   or the natural alignment of their element type when no `.align` appears.
   Lean modeling: this records trusted parameter ABI metadata for the runtime
   buffer decoder, without treating aggregate byte size as its alignment. -/
def signatureParamAlign (decl : String) : Nat := Id.run do
  let toks := ((strim decl).replace "\n" " " |>.replace "\t" " ").splitOn " "
    |>.filter (! ·.isEmpty)
  for i in [0:toks.length] do
    if toks[i]! == ".align" && i + 1 < toks.length then
      return toks[i + 1]!.toNat?.getD 0
  let scalar := toks.foldl (fun n t =>
    if [".b8", ".u8", ".s8"].contains t then 1
    else if [".b16", ".u16", ".s16", ".f16", ".bf16"].contains t then 2
    else if [".b64", ".u64", ".s64", ".f64"].contains t then 8
    else if t == ".b128" then 16 else n) 4
  return scalar * (if toks.contains ".v4" then 4 else if toks.contains ".v2" then 2 else 1)

/- PTX definition: kernel/device functions combine a name, parameter signature, local declarations, and an ordered instruction body. -/
/- Lean modeling: composition preserves the existing signature, locals, aliases, preamble, and translated-statement order byte-for-byte; C ABI and compiler semantics remain trusted. -/
def emitFn (nm sig body : String) (isKernel : Bool) (shTotal : Nat)
    (payloadAnnotated : Bool) : String :=
  let body := scopeRegs (scopeLabels (scopeCallParams body))
  let cSig := if isKernel then "extern \"C\" void " ++ nm ++ "__gxptx(void** args) {"
              else "extern \"C\" __attribute__((visibility(\"hidden\"))) void " ++
                   nm ++ "(void** args) {"
  let locals := regLocals body
  let aliases := signatureParamAliases sig
  let pre := preamble body shTotal
  -- The explicit warp-cohort contract retains lane zero of each warp only.
  -- Election cannot rendezvous with the omitted, effect-free lanes. Keep
  -- ordinary elect.sync unchanged and reject masks electing another lane.
  let election := if isKernel && ((stripComments body).splitOn "gx_ptx_warp_cohort_width").length > 1 then
      "    auto gx_elect_pred = [](uint32_t mask) {\n" ++
      "      if (mask && !(mask & 1u)) gx_ptx_trap_op(\"warp_cohort.elect.mask\");\n" ++
      "      return mask != 0;\n    };\n"
    else ""
  let profileName := nm ++ (if isKernel then "__gxptx" else "")
  let stmts := String.intercalate "\n" (translateBodyScoped body payloadAnnotated profileName)
  cSig ++ "\n" ++ locals ++ aliases ++ pre ++ election ++ stmts ++ "\n}\n"

end Gxptx
