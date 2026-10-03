import Inferred

open Pcv Examples Inferred

private def showFit (name : String) (f : Interface) (names : List String) : IO Unit := do
  let terms := f.terms.zipWith (fun (a, _) n => s!"{a}*{n}") names
  IO.println s!"{name}: {String.intercalate " + " terms} + {f.constant}"

def main : IO Unit := do
  IO.println "Inferred coefficients; each interface has a kernel-checked all-input proof:"
  showFit "linear" linearAuto ["x"]
  showFit "rectangular" rectangularAuto ["(x*y)", "x"]
  showFit "threshold" thresholdAuto ["x", "min(x,y)"]
  showFit "threshold, branch counts" thresholdBranchesAuto ["min(x,y)", "(x-min(x,y))"]
  showFit "threshold, signed" thresholdSignedAuto ["x", "(x-min(x,y))"]
  showFit "triangular" triangularAuto ["(x*(x-1)/2)", "x"]
  showFit "triangular, loop PCV" triangularLoopAuto ["sum(i<x,i)", "x"]
  showFit "fractional" fractionalAuto ["(2*x)"]
  showFit "recomputing cost" cheatAuto ["costProgram(P)"]
  IO.println ""
  IO.println "Threshold, x=100 y=30:"
  let s := inputs 100 30
  IO.println s!"  program: {nodes threshold} nodes; {cost threshold s} operations"
  IO.println s!"  compact PCVs: {thresholdAuto.nodes} nodes; {thresholdAuto.annotationCost s} operations"
  IO.println s!"  recompute PCV: {cheatAuto.nodes} nodes; {cheatAuto.annotationCost s} operations"
  IO.println "Triangular loop, x=100:"
  IO.println s!"  program: {nodes triangular} nodes; {cost triangular s} operations"
  IO.println s!"  closed-form PCVs: {triangularAuto.nodes} nodes; {triangularAuto.annotationCost s} operations"
  IO.println s!"  loop PCVs: {triangularLoopAuto.nodes} nodes; {triangularLoopAuto.annotationCost s} operations"
  IO.println s!"  recompute PCV: {(cheatF triangular).nodes} nodes; {(cheatF triangular).annotationCost s} operations"
  IO.println ""
  IO.println "Raw x,y cannot explain the threshold program (proved for all rational coefficients)."
  IO.println "Compact threshold and triangular interfaces also pass universal annotation-cost budgets."
  IO.println "Costs above count PCV evaluation separately from evaluating the affine formula."
