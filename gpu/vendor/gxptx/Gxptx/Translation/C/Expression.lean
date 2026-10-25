/-
  Rendering of the translator's small C expression AST.

  Integer operations select a finite, semantics-backed `CTemplate`; the
  float/legacy branches retain their historical raw strings.  This separates
  recursive expression traversal from the individual C template spellings.
-/
import Gxptx.Translation.C.Templates
import Gxptx.Semantics.C.Expression

namespace Gxptx

/- PTX definition: these are the modeled integer arithmetic, bitwise, shift, division/remainder, and min/max operation families. -/
/- Lean modeling: `OpK` is only a total selector tag for the existing C templates; it does not introduce a deep PTX instruction syntax. -/
inductive OpK
  | add | sub | mul | mad | neg | bnot | band | bor | bxor
  | shl | lshr | ashr | udiv | sdiv | urem | srem
  | umin | umax | smin | smax
  deriving Repr, DecidableEq

/- PTX definition: every modeled integer operation is interpreted at the instruction's declared 32- or 64-bit width. -/
/- Lean modeling: exhaustive matching routes each operation tag and print width to exactly its corresponding `CTemplate` constructor. -/
def templateOf : OpK → PW → CTemplate
  | .add,  t => .add t
  | .sub,  t => .sub t
  | .mul,  t => .mul t
  | .mad,  t => .mad t
  | .neg,  t => .neg t
  | .bnot, t => .bnot t
  | .band, t => .band t
  | .bor,  t => .bor t
  | .bxor, t => .bxor t
  | .shl,  t => .shl t
  | .lshr, t => .lshr t
  | .ashr, t => .ashr t
  | .udiv, t => .udiv t
  | .sdiv, t => .sdiv t
  | .urem, t => .urem t
  | .srem, t => .srem t
  | .umin, t => .umin t
  | .umax, t => .umax t
  | .smin, t => .smin t
  | .smax, t => .smax t

/- PTX definition: nested operands retain instruction order while each operator uses the declared integer width or legacy float path. -/
/- Lean modeling: recursive rendering is unchanged; integer nodes delegate to `templateOf`/`CTemplate.print`, predicate/select nodes use fixed templates, and trusted float nodes retain byte-identical strings. -/
def CExpr.toC (k : EK := .int .w32) : CExpr → String
  | .var n    => n
  | .lit w    => toString w.toNat ++ "u"
  | .lnot e   => CTemplate.lnot.print (e.toC k) "" ""
  | .bnot e   => match k with
      | .int t => (templateOf .bnot t).print (e.toC k) "" ""
      | .f     => "(~" ++ e.toC k ++ ")"
  | .land a b => CTemplate.land.print (a.toC k) (b.toC k) ""
  | .lor  a b => CTemplate.lor.print (a.toC k) (b.toC k) ""
  | .lxor a b => CTemplate.lxor.print (a.toC k) (b.toC k) ""
  | .band a b => match k with
      | .int t => (templateOf .band t).print (a.toC k) (b.toC k) ""
      | .f     => "(" ++ a.toC k ++ " & " ++ b.toC k ++ ")"
  | .bor  a b => match k with
      | .int t => (templateOf .bor t).print (a.toC k) (b.toC k) ""
      | .f     => "(" ++ a.toC k ++ " | " ++ b.toC k ++ ")"
  | .bxor a b => match k with
      | .int t => (templateOf .bxor t).print (a.toC k) (b.toC k) ""
      | .f     => "(" ++ a.toC k ++ " ^ " ++ b.toC k ++ ")"
  | .add  a b => match k with
      | .int t => (templateOf .add t).print (a.toC k) (b.toC k) ""
      | .f     => "(" ++ a.toC k ++ " + " ++ b.toC k ++ ")"
  | .sub  a b => match k with
      | .int t => (templateOf .sub t).print (a.toC k) (b.toC k) ""
      | .f     => "(" ++ a.toC k ++ " - " ++ b.toC k ++ ")"
  | .mul  a b => match k with
      | .int t => (templateOf .mul t).print (a.toC k) (b.toC k) ""
      | .f     => "(" ++ a.toC k ++ " * " ++ b.toC k ++ ")"
  | .shl  a b => match k with
      | .int t => (templateOf .shl t).print (a.toC k) (b.toC k) ""
      | .f     => "gx_shl(" ++ a.toC k ++ ", " ++ b.toC k ++ ")"
  | .lshr a b => match k with
      | .int t => (templateOf .lshr t).print (a.toC k) (b.toC k) ""
      | .f     => "gx_shr(" ++ a.toC k ++ ", " ++ b.toC k ++ ")"
  | .ashr a b => match k with
      | .int t => (templateOf .ashr t).print (a.toC k) (b.toC k) ""
      | .f     => "gx_sar(" ++ a.toC k ++ ", " ++ b.toC k ++ ")"
  | .neg  a   => match k with
      | .int t => (templateOf .neg t).print (a.toC k) "" ""
      | .f     => "(-" ++ a.toC k ++ ")"
  | .mad  a b c => match k with
      | .int t => (templateOf .mad t).print (a.toC k) (b.toC k) (c.toC k)
      | .f     => "(" ++ a.toC k ++ " * " ++ b.toC k ++ " + " ++ c.toC k ++ ")"
  | .udiv a b => match k with
      | .int t => (templateOf .udiv t).print (a.toC k) (b.toC k) ""
      | .f     => "(" ++ a.toC k ++ " / " ++ b.toC k ++ ")"
  | .sdiv a b => match k with
      | .int t => (templateOf .sdiv t).print (a.toC k) (b.toC k) ""
      | .f     => "(" ++ a.toC k ++ " / " ++ b.toC k ++ ")"
  | .urem a b => match k with
      | .int t => (templateOf .urem t).print (a.toC k) (b.toC k) ""
      | .f     => "gx_urem(" ++ a.toC k ++ ", " ++ b.toC k ++ ")"
  | .srem a b => match k with
      | .int t => (templateOf .srem t).print (a.toC k) (b.toC k) ""
      | .f     => "gx_srem(" ++ a.toC k ++ ", " ++ b.toC k ++ ")"
  | .umin a b => match k with
      | .int t => (templateOf .umin t).print (a.toC k) (b.toC k) ""
      | .f     => "(" ++ a.toC k ++ " < " ++ b.toC k ++ " ? " ++ a.toC k ++ " : " ++ b.toC k ++ ")"
  | .umax a b => match k with
      | .int t => (templateOf .umax t).print (a.toC k) (b.toC k) ""
      | .f     => "(" ++ a.toC k ++ " > " ++ b.toC k ++ " ? " ++ a.toC k ++ " : " ++ b.toC k ++ ")"
  | .smin a b => match k with
      | .int t => (templateOf .smin t).print (a.toC k) (b.toC k) ""
      | .f     => "((int32_t)" ++ a.toC k ++ " < (int32_t)" ++ b.toC k ++ " ? " ++ a.toC k ++ " : " ++ b.toC k ++ ")"
  | .smax a b => match k with
      | .int t => (templateOf .smax t).print (a.toC k) (b.toC k) ""
      | .f     => "((int32_t)" ++ a.toC k ++ " > (int32_t)" ++ b.toC k ++ " ? " ++ a.toC k ++ " : " ++ b.toC k ++ ")"
  | .sel p a b => CTemplate.sel.print (p.toC k) (a.toC k) (b.toC k)

end Gxptx
