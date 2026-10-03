import Pcv

namespace Pcv

/-- Number of free input slots; bound loop indices do not count as inputs. -/
def freeInputs : Expr → Nat → Nat
  | .lit _, _ => 0
  | .var i, depth => i + 1 - depth
  | .add a b, d | .mul a b, d | .sub a b, d | .div a b, d | .seq a b, d =>
      max (freeInputs a d) (freeInputs b d)
  | .iteLt a b t f, d =>
      max (max (freeInputs a d) (freeInputs b d)) (max (freeInputs t d) (freeInputs f d))
  | .sum n b, d => max (freeInputs n d) (freeInputs b (d + 1))

def sampleValues : List Nat := [0, 1, 2, 3, 4, 7, 11]

def sampleInputs : Nat → List (List Nat)
  | 0 => [[]]
  | n + 1 => sampleValues.flatMap fun x => (sampleInputs n).map (x :: ·)

/-- Exact Gaussian elimination. This is proposal generation, not trusted proof.
    Rejects inconsistent and rank-deficient samples rather than inventing a fit. -/
def fitCoefficients (rows : Array (Array Rat)) (width : Nat) : Except String (Array Rat) := do
  let mut matrix := rows
  let mut pivot := 0
  for col in [:width] do
    let mut found : Option Nat := none
    for r in [pivot:matrix.size] do
      if found.isNone && matrix[r]![col]! != 0 then
        found := some r
    let some r := found | throw "insufficient independent variation in the discovery samples"
    let old := matrix[pivot]!
    matrix := matrix.set! pivot matrix[r]!
    matrix := matrix.set! r old
    let divisor := matrix[pivot]![col]!
    let row := matrix[pivot]!.map (· / divisor)
    matrix := matrix.set! pivot row
    for j in [:matrix.size] do
      if j != pivot then
        let factor := matrix[j]![col]!
        matrix := matrix.set! j ((matrix[j]!).zipWith (fun a b => a - factor*b) row)
    pivot := pivot + 1
  for row in matrix do
    if (row.extract 0 width).all (· == 0) && row[width]! != 0 then
      throw "no affine fit on the discovery samples; revise the PCVs"
  return (List.range width).toArray.map fun r => matrix[r]![width]!

def inferInterface (p : Expr) (pcvs : List Expr) : Except String Interface := do
  let arity := pcvs.foldl (fun n e => max n (freeInputs e 0)) (freeInputs p 0)
  if arity > 3 then
    throw "prototype discovery supports at most three free input slots"
  let samples := sampleInputs arity
  let rows := samples.toArray.map fun xs =>
    let s : Env := fun i => xs[i]?.getD 0
    ([1] ++ pcvs.map (fun e => (value e s : Rat)) ++ [(cost p s : Rat)]).toArray
  let coeffs ← fitCoefficients rows (pcvs.length + 1)
  return ⟨coeffs[0]!, pcvs.zipWith (fun e a => (a, e)) (coeffs.toList.drop 1)⟩

open Lean Elab Command

private def ratTerm (q : Rat) : CommandElabM (TSyntax `term) := do
  let magnitude := Syntax.mkNumLit (toString q.num.natAbs)
  let numerator ← if q.num < 0 then `(-$magnitude) else `($magnitude)
  let denominator := Syntax.mkNumLit (toString q.den)
  `(($numerator / $denominator : Rat))

syntax (name := perfDiscover) "perf_discover " ident " : " term " using " "[" term,* "]" : command

/-- Running native code here only proposes coefficients. The generated theorem
    is still proved by perf_check and checked by Lean's kernel. -/
@[command_elab perfDiscover] unsafe def elabPerfDiscover : CommandElab := fun stx => do
  let `(perf_discover $name:ident : $program:term using [$features:term,*]) := stx
    | throwUnsupportedSyntax
  let evaluate (stx : TSyntax `term) : CommandElabM Pcv.Expr := liftTermElabM do
    let e ← Term.elabTermEnsuringType stx (some (mkConst ``Pcv.Expr))
    Term.synthesizeSyntheticMVarsNoPostponing
    let e ← instantiateMVars e
    Meta.evalExpr Pcv.Expr (mkConst ``Pcv.Expr) e
  let p ← evaluate program
  let es ← features.getElems.toList.mapM evaluate
  let fitted ← match inferInterface p es with
    | .ok f => pure f
    | .error message => throwError message
  let c ← ratTerm fitted.constant
  let pairs ← fitted.terms.toArray.zip features.getElems |>.mapM fun ((a, _), e) => do
    let coefficient ← ratTerm a
    `(($coefficient, $e))
  elabCommand (← `(def $name : Pcv.Interface := ⟨$c, [$pairs,*]⟩))
  let proofName := mkIdent (name.getId.appendAfter "_checked")
  elabCommand (← `(theorem $proofName : Pcv.Explains $program $name := by
    unfold $name
    perf_check))
  logInfo m!"{name.getId}: inferred constant {fitted.constant}, coefficients {fitted.terms.map Prod.fst}; generated {proofName.getId}"

end Pcv
