import Examples
import Discover

open Pcv Examples

namespace Inferred

-- These are syntax definitions, not loop invariants or cost theorems.
attribute [local simp] x y light heavy linear rectangular threshold expensive cheap
  triangular pairs pairsByLoop

-- The only user inputs here are the program and its PCVs. No coefficients.
perf_discover linearAuto : linear using [x]
perf_discover rectangularAuto : rectangular using [.mul x y, x]
perf_discover thresholdAuto : threshold using [x, expensive]
perf_discover thresholdBranchesAuto : threshold using [expensive, cheap]
perf_discover thresholdSignedAuto : threshold using [x, cheap]
perf_discover triangularAuto : triangular using [pairs, x]
perf_discover triangularLoopAuto : triangular using [pairsByLoop, x]
perf_discover fractionalAuto : linear using [.mul (.lit 2) x]
perf_discover cheatAuto : threshold using [costProgram threshold]

-- Syntax and evaluation budgets are checked on the inferred interface too.
theorem threshold_budget : Budgeted threshold thresholdAuto 6 1 := by
  unfold thresholdAuto
  perf_check_budget

theorem triangular_budget : Budgeted triangular triangularAuto 8 3 := by
  unfold triangularAuto
  perf_check_budget

#guard match inferInterface threshold [x, y] with | .error _ => true | .ok _ => false
#guard match inferInterface linear [x, x] with | .error _ => true | .ok _ => false
#guard match inferInterface linear [.mul (.lit 2) x] with
  | .ok f => f.constant == 1 && f.terms.map Prod.fst == [(3 : Rat) / 2]
  | .error _ => false

end Inferred
