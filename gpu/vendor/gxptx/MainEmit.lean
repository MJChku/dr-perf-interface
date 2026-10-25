/-
  gxemit — legacy operator-demo CLI backed by the production verified core.

  The previous proof-of-concept duplicated `Ty`, `CExpr`, evaluator functions,
  selectors, and four theorems in this executable root.  The real translator
  later gained a wider production model under `Gxptx`; keeping both copies
  made it unclear which definitions were actually used.  This compatibility
  CLI now calls the production `emitNot`/`emitBin` selectors while retaining
  its historical arguments and exact four demonstrated output shapes.
-/
import Gxptx.Semantics.C.Expression
import Gxptx.Translation.C.Expression

open Gxptx

/- PTX definition: the demo accepts predicate or 32-bit bit/integer forms of `and` and `not`. -/
/- Lean modeling: mapping legacy tokens to the production `Ty` makes the demo exercise the same type-directed selector as the full translator. -/
def parseLegacyTy : String → Option Ty
  | "pred" => some .pred
  | "b32"  => some .b32
  | "s32"  => some .s32
  | "u32"  => some .u32
  | _      => none

/- PTX definition: predicate forms use C logical operators and bit/integer forms use bitwise operators. -/
/- Lean modeling: these four cases preserve the legacy CLI text exactly after the production selector has chosen the correct `CExpr`; every unreachable fallback uses the production printer. -/
def printLegacyDemo : CExpr → String
  | .lnot (.var n) => "(!" ++ n ++ ")"
  | .bnot (.var n) => "(~" ++ n ++ ")"
  | .land (.var a) (.var b) => "(" ++ a ++ " && " ++ b ++ ")"
  | .band (.var a) (.var b) => "(" ++ a ++ " & " ++ b ++ ")"
  | e => e.toC

/- PTX definition: `not` is unary and `and` is binary for the supported scalar/predicate demo forms. -/
/- Lean modeling: argument decoding is only compatibility plumbing; all operator selection is delegated to the production definitions imported above. -/
def main (args : List String) : IO Unit := do
  match args with
  | ["not", ty, r] =>
      match parseLegacyTy ty with
      | some t => IO.println (printLegacyDemo (emitNot t (.var r)))
      | none   => IO.eprintln s!"unknown type {ty}"
  | ["and", ty, a, b] =>
      match parseLegacyTy ty with
      | some t => IO.println (printLegacyDemo (emitBin t .and (.var a) (.var b)))
      | none   => IO.eprintln s!"unknown type {ty}"
  | _ => IO.eprintln "usage: gxemit not|and <pred|b32> <regs...>"
