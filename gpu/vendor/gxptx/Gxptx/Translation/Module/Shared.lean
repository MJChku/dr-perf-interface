/-
  Gxptx.Translation.Module.Shared — PTX image boundaries, shared-memory
  declarations and layout, plus the per-function special-register preamble.
-/
import Gxptx.Translation.Parsing
import Std.Data.HashSet
set_option linter.deprecated false
namespace Gxptx

/- PTX definition: shared scalars occupy their type width; array extents count elements, while extern shared arrays may have an unsized extent. -/
/- Lean modeling: this shallow tuple retains exactly the three fields needed by GX layout and uses `none` to distinguish a dynamic extern allocation from a zero-byte static object. -/
abbrev SharedDecl := String × Option Nat × Nat

/- PTX definition: a variable's address must satisfy its declared byte alignment. -/
/- Lean modeling: ceiling division computes the least aligned offset not below `n`, with zero treated as no additional constraint; this is GX layout arithmetic, not a PTX address-space theorem. -/
def alignUp (n a : Nat) : Nat :=
  if a == 0 then n else ((n + a - 1) / a) * a

/- PTX definition: every PTX module begins with a required `.version` directive. -/
/- Lean modeling: treating each post-marker segment as an independent image prevents symbols and shared offsets from leaking across fatbin modules; disassembly framing remains trusted parsing. -/
def ptxModules (src : String) : Array String :=
  match src.splitOn ".version" with
  | [_] => #[src]
  | _prefix :: modules => modules.toArray
  | [] => #[src]

/- PTX definition: `.shared` declarations specify identifiers, optional array extents, and optional `.align` constraints. -/
/- Lean modeling: token scanning preserves the first declaration of each symbol and distinguishes sized/static from unsized/extern storage; completeness of permissive text parsing is trusted. -/
def sharedDecls (src : String) : Array SharedDecl := Id.run do
  let mut out : Array SharedDecl := #[]
  let mut seen : Std.HashSet String := {}
  for line in (src.splitOn "\n") do
    let l := strim ((line.splitOn "//").headD "" |>.replace "\t" " ")
    if (l.startsWith ".shared" || l.startsWith ".visible .shared"
        || l.startsWith ".extern .shared") && (l.splitOn "%").length == 1 then
      let toks := (l.splitOn " ").filter (! ·.isEmpty)
      match toks.reverse with
      | last :: _ =>
        let spec := last.replace ";" ""
        -- PTX array extents count elements, not bytes. Scalars are static
        -- objects too; only an explicitly unsized array uses the dynamic base.
        let scalarBytes := toks.foldl (fun bytes tok =>
          match tok with
          | ".b8" | ".u8" | ".s8" => 1
          | ".b16" | ".u16" | ".s16" | ".f16" | ".bf16" => 2
          | ".b32" | ".u32" | ".s32" | ".f32" => 4
          | ".b64" | ".u64" | ".s64" | ".f64" => 8
          | ".b128" => 16
          | _ => bytes) 1
        let vectorWidth := if toks.contains ".v4" then 4 else if toks.contains ".v2" then 2 else 1
        let elementBytes := scalarBytes * vectorWidth
        let align := Id.run do
          let mut a := elementBytes
          for i in [0:toks.length] do
            if toks[i]! == ".align" && i + 1 < toks.length then
              a := toks[i + 1]!.toNat?.getD 1
          return a
        match (spec.replace "]" "").splitOn "[" with
        | [nm, szs] =>
          if ! seen.contains nm then
            seen := seen.insert nm
            let size := if szs.isEmpty then none else some ((szs.toNat?.getD 0) * elementBytes)
            out := out.push (nm, size, align)
        | [nm] =>
          if ! seen.contains nm then
            seen := seen.insert nm
            out := out.push (nm, some elementBytes, align)
        | _ => pure ()
      | _ => pure ()
  return out

/- PTX definition: shared objects occupy aligned addresses, and unsized extern shared declarations refer to runtime-sized storage. -/
/- Lean modeling: laying out all fixed objects before the dynamic base avoids source-order overlap and respects declared alignment. GX also aligns an extern region to at least 1024 bytes: unrelated kernels' fixed objects share this module layout, and must not introduce low address bits into position-independent 128-byte shared swizzles. This over-alignment and the alias policy are GX ABI trust. -/
def sharedLayout (src : String) : Nat × Array (String × Nat) := Id.run do
  let decls := sharedDecls src
  let mut out : Array (String × Nat) := #[]
  let mut off := 0
  for (nm, size, align) in decls do
    match size with
    | some n =>
      off := alignUp off (max 16 align)
      out := out.push (nm, off)
      off := off + n
    | none => pure ()
  let mut dynamicAlign := 1
  for (_, size, align) in decls do
    if size.isNone then dynamicAlign := max dynamicAlign (max 1024 align)
  let dynamicBase := alignUp (alignUp off 16) dynamicAlign
  for (nm, size, _) in decls do
    if size.isNone then out := out.push (nm, dynamicBase)
  return (dynamicBase, out)

/- PTX definition: statically sized and dynamically sized shared objects inhabit a kernel's shared state space. -/
/- Lean modeling: projecting the base computed by `sharedLayout` keeps declaration parsing and preamble allocation on one layout result; the host arena representation remains trusted. -/
def sharedTotal (src : String) : Nat := (sharedLayout src).1

/- PTX definition: functions may read shared storage and thread/block special registers. -/
/- Lean modeling: source-use tests avoid unused declarations while existing runtime accessors initialize exactly the names produced by `sreg?`; accessor and arena correctness remain trusted. -/
def preamble (body : String) (shTotal : Nat) : String :=
  let needTid := ["%tid", "%ntid", "%ctaid", "%nctaid", "_gx_tid", "_gx_ntid", "_gx_ctaid", "_gx_nctaid"].any
    (fun token => (body.splitOn token).length > 1)
  let needShm := (body.splitOn "_gx_shm").length > 1 || (body.splitOn "shared").length > 1 ||
    (body.splitOn "wgmma.").length > 1
  (if needTid then
      "    dim3 _gx_tid = gx_dev_threadIdx();\n" ++
      "    dim3 _gx_ntid = gx_dev_blockDim();\n" ++
      "    dim3 _gx_ctaid = gx_dev_blockIdx();\n" ++
      "    dim3 _gx_nctaid = gx_dev_gridDim();\n"
    else "") ++
  (if needShm then "    char* _gx_shm = (char*)gx_ptx_shared_window(" ++ toString shTotal ++ "u);\n" else "")

end Gxptx
