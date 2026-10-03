import Pcv
open Pcv

namespace Examples

def x : Expr := .var 0
def y : Expr := .var 1

def light : Expr := .add (.lit 0) (.lit 0)
def heavy : Expr := .add light light

def linear : Expr := .sum x light
def linearF : Interface := ⟨1, [(3, x)]⟩

def rectangular : Expr := .sum x (.sum (.var 2) light)
def rectangularF : Interface := ⟨1, [(3, .mul x y), (3, x)]⟩

def threshold : Expr := .sum x (.iteLt (.var 0) (.var 2) heavy light)
def expensive : Expr := .iteLt x y x y
def cheap : Expr := .sub x expensive
def thresholdF : Interface := ⟨1, [(6, expensive), (4, cheap)]⟩

def thresholdCompactF : Interface := ⟨1, [(4, x), (2, expensive)]⟩

def triangular : Expr := .sum x (.sum (.var 0) light)
def pairs : Expr := .div (.mul x (.sub x (.lit 1))) (.lit 2)
def triangularF : Interface := ⟨1, [(3, pairs), (3, x)]⟩

-- PCVs really may contain loops. This one replaces a quadratic loop with
-- a linear-time sum; the closed-form version above removes its loop too.
def pairsByLoop : Expr := .sum x (.var 0)
def triangularLoopF : Interface := ⟨1, [(3, pairsByLoop), (3, x)]⟩

def cheatF (p : Expr) : Interface := ⟨0, [(1, costProgram p)]⟩

attribute [local simp] x y light heavy linear linearF rectangular rectangularF
  threshold thresholdF expensive cheap
  thresholdCompactF triangular pairs triangularF pairsByLoop triangularLoopF cheatF

theorem linear_checked : Explains linear linearF := by perf_check

theorem rectangular_checked : Explains rectangular rectangularF := by perf_check

theorem threshold_checked : Explains threshold thresholdF := by perf_check

theorem threshold_compact_checked : Explains threshold thresholdCompactF := by perf_check

theorem triangular_checked : Explains triangular triangularF := by perf_check

theorem triangular_loop_pcv_checked : Explains triangular triangularLoopF := by perf_check

theorem cheating_always_checks (p : Expr) : Explains p (cheatF p) := by perf_check

def inputs (x y : Nat) : Env
  | 0 => x
  | 1 => y
  | _ => 0

/-- Failure of raw x,y is mathematical, not merely a limitation of the tactic.
    Four concrete states refute EVERY choice of signed affine coefficients. -/
theorem threshold_not_raw :
    ¬ ∃ a b d : Rat, Explains threshold ⟨d, [(a, x), (b, y)]⟩ := by
  rintro ⟨a, b, d, h⟩
  have h00 := h (inputs 0 0)
  have h01 := h (inputs 0 1)
  have h10 := h (inputs 1 0)
  have h11 := h (inputs 1 1)
  simp [Interface.eval, cost, value, push, inputs, sumRange] at h00 h01 h10 h11
  grind

/-- Negative regression: automation must not accept an incorrect coefficient. -/
example : True := by
  fail_if_success
    have _bad : Explains linear ⟨1, [(4, x)]⟩ := by perf_check
  trivial

/-- PCV evaluation is explicitly distinguished from evaluating the affine formula. -/
theorem threshold_annotation_cost (s : Env) :
    thresholdCompactF.annotationCost s = 1 := by
  simp [Interface.annotationCost, cost]

theorem triangular_annotation_cost (s : Env) :
    triangularF.annotationCost s = 3 := by
  simp [Interface.annotationCost, cost]

theorem triangular_loop_annotation_cost (s : Env) :
    triangularLoopF.annotationCost s = 2 * s 0 + 1 := by
  simp [Interface.annotationCost, cost, value]
  omega

theorem threshold_budgeted : Budgeted threshold thresholdCompactF 6 1 := by
  perf_check_budget

theorem triangular_budgeted : Budgeted triangular triangularF 8 3 := by
  perf_check_budget

example : True := by
  fail_if_success
    have _expensive : Budgeted threshold (cheatF threshold) 6 1 := by
      perf_check_budget
  trivial

/-- A concrete input also proves the cheat violates the runtime budget,
    even if we remove the syntax-size obstacle. -/
theorem cheat_exceeds_runtime_budget :
    ¬ ∀ s, (cheatF threshold).annotationCost s ≤ 1 := by
  intro h
  have bad := h (inputs 1 0)
  have measured : (cheatF threshold).annotationCost (inputs 1 0) > 1 := by decide
  omega

#guard (cost threshold (inputs 100 30), thresholdCompactF.annotationCost (inputs 100 30))
    == (461, 1)
#guard (cost triangular (inputs 100 0), triangularF.annotationCost (inputs 100 0),
        triangularLoopF.annotationCost (inputs 100 0)) == (15151, 3, 201)

end Examples
