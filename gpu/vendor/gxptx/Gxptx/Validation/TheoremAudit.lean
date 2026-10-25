/-
  Environment discovery for the whole-project theorem audit.

  Declarations are selected by the source module that contributed their
  `ConstantInfo`, not by their generated name.  This includes private theorem
  names such as `_private.Gxptx.Verification....` in the runtime audit.
-/
import Lean

open Lean

namespace Gxptx

/-- True exactly for production modules in the `Gxptx` namespace. -/
/- PTX definition: PTX does not define this module-filter helper; it is project validation infrastructure. -/
/- Lean modeling: filtering by contributing module includes private and generated theorem names that name-prefix heuristics could miss. -/
def isGxptxModule (n : Name) : Bool :=
  let s := n.toString
  s == "Gxptx" || s.startsWith "Gxptx."

/-- Every unique theorem contributed by an imported `Gxptx.*` module. -/
/- PTX definition: PTX does not define theorem discovery; the discovered propositions encode project ISA obligations. -/
/- Lean modeling: module provenance and ConstantInfo.isTheorem cover public, private, mangled, and generated theorem declarations uniformly. -/
def allGxptxTheorems (env : Environment) : List Name := Id.run do
  let mut out : List Name := []
  for moduleIdx in [0:env.header.moduleNames.size] do
    let moduleName := env.header.moduleNames[moduleIdx]!
    if isGxptxModule moduleName then
      match env.header.moduleData[moduleIdx]? with
      | none => pure ()
      | some moduleData =>
        for cinfo in moduleData.constants do
          if cinfo.isTheorem && !out.contains cinfo.name then
            out := cinfo.name :: out
  return out.reverse

/-- Number of imported production modules selected by `isGxptxModule`. -/
/- PTX definition: PTX does not define this module-count diagnostic; it checks project audit completeness. -/
/- Lean modeling: reporting the count makes unexpected import shrinkage visible alongside the theorem total. -/
def gxptxModuleCount (env : Environment) : Nat :=
  env.header.moduleNames.foldl
    (fun count n => if isGxptxModule n then count + 1 else count) 0

end Gxptx
