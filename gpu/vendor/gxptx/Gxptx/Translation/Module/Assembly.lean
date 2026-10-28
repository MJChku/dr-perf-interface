/-
  Gxptx.Translation.Module.Assembly — module-scope shared/global declarations,
  forward declarations, kernel metadata, and complete or chunked C assembly.
-/
import Gxptx.Translation.Module.Functions
set_option linter.deprecated false
namespace Gxptx

/- PTX definition: a shared variable's address is a shared-window address of the executing CTA. -/
/- Lean modeling: the offset constant is tagged at each use with the CTA's cluster-rank bits, so symbol and `cvta.to.shared` addresses agree. -/
def sharedSymbolDef (nm : String) (off : Nat) : String :=
  s!"static const unsigned {nm}__gxoff = {off}u;\n#define {nm} ({nm}__gxoff | gx_shared_rank_bits())\n"

/- PTX definition: shared variables occupy addresses in a per-CTA shared state space. -/
/- Lean modeling: each shared symbol names its `sharedLayout` byte offset tagged with the executing CTA's cluster rank in bits [24,32), matching hardware shared::cta addresses and GX's masked `_gx_shm + offset` access; the host shared window and rank accessor remain `trusted:cluster-runtime`. -/
def sharedConsts (src : String) : String := Id.run do
  let mut out : Array String := #[]
  for (nm, off) in (sharedLayout src).2 do
    out := out.push (sharedSymbolDef nm off)
  return String.join out.toList

/- PTX definition: shared storage belongs to an individual versioned PTX module. -/
/- Lean modeling: per-image layout prevents cross-module offsets and first-definition filtering matches function deduplication; fatbin concatenation remains trusted parsing. -/
def fatbinSharedConsts (src : String) : String := Id.run do
  let mut out : Array String := #[]
  let mut seen : Std.HashSet String := {}
  for image in ptxModules src do
    for (nm, off) in (sharedLayout image).2 do
      if ! seen.contains nm then
        seen := seen.insert nm
        out := out.push (sharedSymbolDef nm off)
  return String.join out.toList

/- PTX definition: PTX modules declare named kernel entry points and device functions before or alongside calls/references. -/
/- Lean modeling: preserving each extracted name and kernel flag allows cross-calls/function tables independent of C definition order; visibility and ABI remain trusted. -/
def forwardDecls (fns : Array (String × String × String × Bool × Nat)) : String :=
  String.intercalate "\n" (fns.toList.map (fun (nm, _, _, k, _) =>
    if k then s!"extern \"C\" void {nm}__gxptx(void**);"
    else s!"extern \"C\" __attribute__((visibility(\"hidden\"))) void {nm}(void**);")) ++ "\n"

/- PTX definition: an uninitialized constant/global scalar or fixed-size array
   still defines storage; array extents count elements, not bytes.
   Lean modeling: aligned byte storage preserves its address, extent and zero
   initialization. Constant-space writes by the host remain a runtime operation;
   the generated host storage must therefore not be C++ `const`. -/
def uninitializedModuleGlobal (line : String) : Option (String × Nat × Nat) := do
  let line := strim (line.replace "\n" " " |>.replace "\r" " " |>.replace "\t" " ")
  if !line.endsWith ";" || (line.splitOn "=").length != 1 then none else do
    let (decl, count) ← match (line.dropRight 1).splitOn "[" with
      | [decl] => some (decl, 1)
      | [decl, tail] => match tail.splitOn "]" with
        | [extent, rest] =>
          if !(strim rest).isEmpty then none
          else (strim extent).toNat?.map (decl, ·)
        | _ => none
      | _ => none
    let toks := (decl.splitOn " ").filter (! ·.isEmpty)
    let scalar := toks.foldl (fun bytes tok =>
      match tok with
      | ".b8" | ".u8" | ".s8" => 1
      | ".b16" | ".u16" | ".s16" | ".f16" | ".bf16" => 2
      | ".b32" | ".u32" | ".s32" | ".f32" => 4
      | ".b64" | ".u64" | ".s64" | ".f64" => 8
      | ".b128" => 16
      | _ => bytes) 0
    let elem := scalar * (if toks.contains ".v4" then 4 else if toks.contains ".v2" then 2 else 1)
    let name := toks.getLastD ""
    let alignment := signatureParamAlign line
    if elem == 0 || count == 0 || name.isEmpty || alignment == 0 then none
    else some (name, elem * count, alignment)

private def startsModuleGlobal (text : String) : Bool :=
  let tokens := (text.splitOn " ").filter (! ·.isEmpty)
  match tokens with
  | ".global" :: _ | ".const" :: _ => true
  | ".visible" :: ".global" :: _ | ".visible" :: ".const" :: _ => true
  | _ => false

/- Recognize only uninitialized module declarations. Semicolons in quoted PTX
   strings/comments are not declaration boundaries; braces exclude function
   bodies and initialized data. Ending line numbers let the existing initialized
   parser retain its ordering and behavior. -/
def uninitializedModuleDeclarations (src : String) : Array (Nat × String) := Id.run do
  let chars := src.toList.toArray
  let mut result := #[]
  let mut current := ""
  let mut line := 0
  let mut depth := 0
  let mut quoted := false
  let mut escaped := false
  let mut lineComment := false
  let mut blockComment := false
  let mut i := 0
  while i < chars.size do
    let c := chars[i]!
    let next := chars[i + 1]?
    let sourceLine := line
    if c == '\n' then line := line + 1
    if lineComment then
      if c == '\n' then
        lineComment := false
        let text := strim current
        current := if depth == 0 && (text == ".visible" || startsModuleGlobal text)
          then text ++ " " else ""
    else if blockComment then
      if c == '*' && next == some '/' then
        blockComment := false
        i := i + 1
    else if quoted then
      current := current.push c
      if escaped then escaped := false
      else if c == '\\' then escaped := true
      else if c == '"' then quoted := false
    else if c == '/' && next == some '/' then
      lineComment := true
      i := i + 1
    else if c == '/' && next == some '*' then
      blockComment := true
      current := current ++ " "
      i := i + 1
    else if c == '"' then
      quoted := true
      current := current.push c
    else if c == '{' then
      depth := depth + 1
      current := ""
    else if c == '}' then
      depth := depth - 1
      current := ""
    else if depth == 0 then
      if c == ';' then
        let decl := strim current ++ ";"
        if startsModuleGlobal (strim current) && (uninitializedModuleGlobal decl).isSome then
          result := result.push (sourceLine, decl)
        current := ""
      else if c == '\n' then
        let text := strim current
        current := if text == ".visible" || startsModuleGlobal text then text ++ " " else ""
      else if c.isWhitespace then current := current ++ " "
      else current := current.push c
    i := i + 1
  return result

/- PTX definition: module variables may occupy constant/global state space and use bounded array initializers containing data or symbol addresses. -/
/- Lean modeling: extent preservation keeps implicit trailing zeros, symbol entries become pointer-width integers, and chunk-zero ownership avoids duplicate definitions; linkage/host layout stay trusted. -/
def moduleGlobals (src : String) : String × String := Id.run do
  let mut defs : Array String := #[]
  let mut externs : Array String := #[]
  let mut seen : Std.HashSet String := {}
  let mut pending := (uninitializedModuleDeclarations src).toList
  for (lineNumber, line) in (src.splitOn "\n").enum do
    let l := strim ((line.splitOn "//").headD "" |>.replace "\t" " ")
    let declarations := pending.takeWhile (fun entry => entry.1 == lineNumber)
    pending := pending.drop declarations.length
    for (_, declaration) in declarations do
      if let some (key, bytes, alignment) := uninitializedModuleGlobal declaration then
        if !seen.contains key then
          seen := seen.insert key
          defs := defs.push ("extern \"C\" { " ++ s!"alignas({alignment}) uint8_t {key}[{bytes}] = " ++ "{}; }\n")
          externs := externs.push s!"extern \"C\" uint8_t {key}[{bytes}];\n"
    if (l.startsWith ".global" || l.startsWith ".const" || l.startsWith ".visible .global"
        || l.startsWith ".visible .const") && (l.splitOn "= {").length > 1 then
      match (l.splitOn "= {") with
      | decl :: rhs :: _ =>
        let nm := ((decl.splitOn " ").filter (! ·.isEmpty)).getLastD "T"
        let key := (nm.splitOn "[").headD nm
        -- Keep the PTX extent. An initializer may omit trailing zero values;
        -- C only supplies those implicit zeros when the declared array bound
        -- is preserved. CUDA format strings rely on that trailing NUL.
        let extent := match (nm.replace "]" "").splitOn "[" with
          | _ :: n :: _ => n
          | _ => ""
        let bound := if extent.isEmpty then "[]" else s!"[{extent}]"
        if seen.contains key then pure () else
          seen := seen.insert key
          let entries := ((rhs.replace "}" "").replace ";" "").splitOn ","
          let firstEntry := strim (entries.headD "")
          let isFn := ! firstEntry.isEmpty && ((firstEntry.get 0).isAlpha || firstEntry.get 0 == '_')
          if isFn then
            let vals := entries.map (fun e =>
              let e := strim e
              if e.isEmpty then "0"
              else if (e.get 0).isAlpha || e.get 0 == '_' then s!"(uint64_t)(uintptr_t)&{e}"
              else s!"(uint64_t)({e}ull)")
            defs := defs.push (s!"extern \"C\" uint64_t {key}{bound} = " ++ "{" ++
                    String.intercalate ", " vals ++ "};\n")
            externs := externs.push s!"extern \"C\" uint64_t {key}{bound};\n"
          else
            let vals := entries.map strim
            defs := defs.push (s!"extern \"C\" uint8_t {key}{bound} = " ++ "{" ++
                    String.intercalate ", " vals ++ "};\n")
            externs := externs.push s!"extern \"C\" uint8_t {key}{bound};\n"
      | _ => pure ()
  return (String.join defs.toList, String.join externs.toList)

/- PTX definition: a kernel entry point has an ordered parameter signature and an instruction body. -/
/- Lean modeling: sizes follow `signatureParamSizes` order and body-text tests preserve execution-mode metadata; mode 2 makes every block resident for cooperative-grid synchronization, while modes 0/1 retain fiber/syncfree execution. The heuristic and GX launch ABI are trusted. -/
def kernelMeta (fns : Array (String × String × String × Bool × Nat)) : String :=
  let gridFns : Std.HashSet String := Id.run do
    let mut result : Std.HashSet String := {}
    for (nm, _, body, _, _) in fns do
      if ["%envreg1", "%envreg2", "gx_ptx_grid_sync"].any
          (fun marker => (body.splitOn marker).length > 1) then
        result := result.insert nm
    -- A device helper can contain the rendezvous. Its callers need the same
    -- all-resident guarantee; checking only the entry body can deadlock.
    for _ in [0:fns.size] do
      let oldSize := result.size
      for (nm, _, body, _, _) in fns do
        let words := ((((stripComments body).replace "," " ").replace ";" " ").replace "\n" " ").replace "\t" " " |>.splitOn " "
        if !result.contains nm && words.any result.contains then
          result := result.insert nm
      if result.size == oldSize then break
    return result
  String.intercalate "\n" (fns.toList.filterMap (fun (nm, sig, body, k, _) =>
    if k then
      let sizes := signatureParamSizes sig
      let aligns := (signatureParamDecls sig).map signatureParamAlign
      let gridSync := gridFns.contains nm
      -- An ordinary launch can implement an inter-CTA wait in global memory
      -- without cooperative_groups / envreg. Batching such blocks can keep
      -- the producer unlaunched forever. Conservatively admit the full grid
      -- when a strong global/generic load coexists with a backward edge.
      -- This may admit extra blocks for a payload loop; it never removes work.
      let words := ((stripComments body).replace "\n" " ").replace "\t" " " |>.splitOn " "
      let strongLoad := words.any (fun op =>
        let ts := op.splitOn "."
        ts.headD "" == "ld" &&
        ["volatile", "acquire", "relaxed"].any ts.contains &&
        !ts.any (fun t => t.startsWith "shared" || ["local", "param", "const"].contains t))
      -- A poll annotation hides the load inside a native device helper. It
      -- still waits on another CTA: never batch away its producer blocks.
      let annotatedPoll := ["gx_ptx_control_poll", "gx_ptx_control_load_acquire_sys_u32", "gx_ptx_control_load_sys_u32", "gx_ptx_poll_sys_u32", "gx_ptx_poll_sys_u64", "gx_ptx_poll_gpu_u64"].any
        (fun marker => (body.splitOn marker).length > 1)
      let globalWait := (strongLoad || annotatedPoll) && (translateBody (scopeLabels body)).any
        (fun line => line == "    gx_ptx_spin_yield();")
      -- Warp exchanges rendezvous just like block barriers. A call may hide
      -- either kind of synchronization in a device helper.
      let fibers := ["bar", "shfl.", "vote.", "redux.", "elect.", "match.", "ldmatrix.", "stmatrix.", "mma.", "tcgen05.", "setmaxnreg.", "call"].any
        (fun token => (body.splitOn token).length > 1)
      -- Stronger explicit contract: one representative per block and no
      -- inter-block dependencies. Such blocks can execute sequentially on
      -- one reused context, without allocating a CUDA fiber for each block.
      let independent := ((stripComments body).splitOn "gx_ptx_independent_block_cohort_width").length > 1
      let mode := if gridSync || globalWait then "2" else if independent then "1" else if fibers then "0" else "1"
      -- Explicit annotation contract: x!=0 lanes have no effects beyond the
      -- representative's counted cooperative operations. This is NOT inferred
      -- from a kernel name or propagated from arbitrary device helpers.
      let blockCohort := independent || ((stripComments body).splitOn "gx_ptx_block_cohort_width").length > 1
      let warpCohort := ((stripComments body).splitOn "gx_ptx_warp_cohort_width").length > 1
      some (s!"extern \"C\" const unsigned {nm}__gxptx_nparams = {sizes.length};\n" ++
            s!"extern \"C\" const unsigned {nm}__gxptx_footprint_abi = 1;\n" ++
            s!"extern \"C\" const unsigned {nm}__gxptx_latency_abi = 1;\n" ++
            s!"extern \"C\" const unsigned {nm}__gxptx_psizes[] = " ++ "{ " ++
            String.intercalate ", " (sizes.map toString) ++ " };\n" ++
            s!"extern \"C\" const unsigned {nm}__gxptx_paligns[] = " ++ "{ " ++
            String.intercalate ", " (aligns.map toString) ++ " };\n" ++
            s!"extern \"C\" const unsigned {nm}__gxptx_syncfree = {mode};" ++
            if blockCohort then s!"\nextern \"C\" const unsigned {nm}__gxptx_block_cohort = 1;"
            else if warpCohort then s!"\nextern \"C\" const unsigned {nm}__gxptx_block_cohort = 2;" else "")
    else none))

/- PTX definition: a module contains directives, globals, kernel entries, and device functions whose references share module scope. -/
/- Lean modeling: every chunk receives common declarations, only chunk zero owns definitions/metadata, and function bodies retain individual translation order; splitting is build plumbing. -/
def translateModuleChunks (prelude src : String) (nsplit : Nat) : Array String := Id.run do
  let fns := extractFns src
  let payloadAnnotated := (src.splitOn "gx_ptx_payload_scope").length > 1 ||
                          (src.splitOn "gx_ptx_payload_event").length > 1
  let fwd := forwardDecls fns
  let shared := fatbinSharedConsts src
  let (defs, externs) := moduleGlobals src
  let meta := kernelMeta fns
  let mut chunks : Array String := Array.mkArray nsplit ""
  -- chunk 0 defines globals; other chunks get extern decls. shared consts +
  -- forward decls in every chunk.
  for i in [0:nsplit] do
    let hdr := if i == 0 then prelude ++ "\n" ++ shared ++ "\n" ++ fwd ++ "\n" ++ defs ++ "\n"
               else prelude ++ "\n" ++ shared ++ "\n" ++ fwd ++ "\n" ++ externs ++ "\n"
    chunks := chunks.set! i hdr
  -- bodies round-robin
  let mut idx := 0
  for (nm, sig, body, k, shTotal) in fns do
    let c := idx % nsplit
    chunks := chunks.set! c (chunks[c]! ++ emitFn nm sig body k shTotal payloadAnnotated ++ "\n")
    idx := idx + 1
  -- metadata into chunk 0
  chunks := chunks.set! 0 (chunks[0]! ++ meta ++ "\n")
  return chunks

/- PTX definition: a PTX module combines module directives, variables, entry points, and functions. -/
/- Lean modeling: concatenation preserves the established prelude/declaration/global/function/metadata order; parsing, C linkage, runtime ABI, and compiler behavior remain trusted boundaries. -/
def translateModule (prelude src : String) : String :=
  let fns := extractFns src
  let payloadAnnotated := (src.splitOn "gx_ptx_payload_scope").length > 1 ||
                          (src.splitOn "gx_ptx_payload_event").length > 1
  prelude ++ "\n" ++ fatbinSharedConsts src ++ "\n" ++ forwardDecls fns ++ "\n"
    ++ (moduleGlobals src).1 ++ "\n"
    ++ String.intercalate "\n" (fns.toList.map (fun (nm, sig, body, k, shTotal) =>
      emitFn nm sig body k shTotal payloadAnnotated))
    ++ "\n" ++ kernelMeta fns ++ "\n"

end Gxptx
