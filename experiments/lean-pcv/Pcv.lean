import Lean

namespace Pcv

/-- A single inlined language for both measured programs and PCVs.
    `sum n body` binds variable 0 to the iteration index and shifts outer variables.
    There are no function calls, arbitrary host functions, or cost-query primitives. -/
inductive Expr where
  | lit : Nat → Expr
  | var : Nat → Expr
  | add : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | seq : Expr → Expr → Expr
  | iteLt : Expr → Expr → Expr → Expr → Expr
  | sum : Expr → Expr → Expr
  deriving Repr, DecidableEq

abbrev Env := Nat → Nat

def push (s : Env) (i : Nat) : Env
  | 0 => i
  | j + 1 => s j

def sumRange : Nat → (Nat → Nat) → Nat
  | 0, _ => 0
  | n + 1, f => sumRange n f + f n

/-- Return value. Loops sum the values returned by their bodies. -/
def value : Expr → Env → Nat
  | .lit n, _ => n
  | .var x, s => s x
  | .add a b, s => value a s + value b s
  | .mul a b, s => value a s * value b s
  | .sub a b, s => value a s - value b s
  | .div a b, s => value a s / value b s
  | .seq _ b, s => value b s
  | .iteLt a b t f, s => if value a s < value b s then value t s else value f s
  | .sum n b, s => sumRange (value n s) (fun i => value b (push s i))

/-- Unit-cost operational model: literals/reads and sequencing cost zero;
    arithmetic and a comparison cost one. A loop charges one final test,
    and one test plus one accumulation per iteration, in addition to its
    bound evaluation and body evaluation. This is an abstract source cost. -/
def cost : Expr → Env → Nat
  | .lit _, _ => 0
  | .var _, _ => 0
  | .add a b, s | .mul a b, s | .sub a b, s | .div a b, s => cost a s + cost b s + 1
  | .seq a b, s => cost a s + cost b s
  | .iteLt a b t f, s => cost a s + cost b s + 1 +
      if value a s < value b s then cost t s else cost f s
  | .sum n b, s => cost n s + 1 +
      sumRange (value n s) (fun i => cost b (push s i) + 2)

/-- Exact rational affine coefficients; the program and PCVs return naturals. -/
structure Interface where
  constant : Rat
  terms : List (Rat × Expr)
  deriving Repr

def Interface.eval (f : Interface) (s : Env) : Rat :=
  f.constant + (f.terms.map fun (a, e) => a * (value e s : Rat)).sum

/-- Every input, not a finite sample. PCVs are evaluated on the entry environment. -/
def Explains (p : Expr) (f : Interface) : Prop :=
  ∀ s, (cost p s : Rat) = f.eval s

@[simp] theorem sumRange_const (n a : Nat) : sumRange n (fun _ => a) = n * a := by
  induction n with
  | zero => simp [sumRange]
  | succ n ih => simp [sumRange, ih, Nat.succ_mul]

@[simp] theorem sumRange_add (n : Nat) (f g : Nat → Nat) :
    sumRange n (fun i => f i + g i) = sumRange n f + sumRange n g := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sumRange, ih]; omega

@[simp] theorem sumRange_mul_right (n : Nat) (f : Nat → Nat) (a : Nat) :
    sumRange n (fun i => f i * a) = sumRange n f * a := by
  induction n with
  | zero => simp [sumRange]
  | succ n ih => simp [sumRange, ih, Nat.add_mul]

@[simp] theorem sumRange_mul_left (n : Nat) (a : Nat) (f : Nat → Nat) :
    sumRange n (fun i => a * f i) = a * sumRange n f := by
  simp only [Nat.mul_comm a, sumRange_mul_right]

theorem sumRange_index (n : Nat) :
    sumRange n (fun i => i) = n * (n - 1) / 2 := by
  have doubled : ∀ n, 2 * sumRange n (fun i => i) = n * (n - 1) := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      simp only [sumRange, Nat.mul_add, ih]
      cases n with
      | zero => rfl
      | succ n => simp [Nat.add_mul, Nat.mul_add]; grind
  have h := doubled n
  omega

/-- Generic threshold-branch lemma, proved once in the checker library. -/
theorem sumRange_threshold (n y a b : Nat) :
    sumRange n (fun i => if i < y then a else b) =
      min n y * a + (n - y) * b := by
  induction n with
  | zero => simp [sumRange]
  | succ n ih =>
    rw [sumRange, ih]
    by_cases h : n < y
    · have hn : min n y = n := Nat.min_eq_left (by omega)
      have hs : min (n + 1) y = n + 1 := Nat.min_eq_left (by omega)
      simp [h, hn, hs, Nat.sub_eq_zero_of_le (show n ≤ y by omega),
        Nat.sub_eq_zero_of_le (show n + 1 ≤ y by omega), Nat.succ_mul]
    · have hn : min n y = y := Nat.min_eq_right (by omega)
      have hs : min (n + 1) y = y := Nat.min_eq_right (by omega)
      have hd : n + 1 - y = (n - y) + 1 := by omega
      simp [h, hn, hs, hd, Nat.add_mul, Nat.add_assoc]

/-- Count each expression node, including literals and variable occurrences. -/
def nodes : Expr → Nat
  | .lit _ | .var _ => 1
  | .add a b | .mul a b | .sub a b | .div a b | .seq a b | .sum a b => 1 + nodes a + nodes b
  | .iteLt a b t f => 1 + nodes a + nodes b + nodes t + nodes f

def loopDepth : Expr → Nat
  | .lit _ | .var _ => 0
  | .add a b | .mul a b | .sub a b | .div a b | .seq a b => max (loopDepth a) (loopDepth b)
  | .iteLt a b t f => max (max (loopDepth a) (loopDepth b)) (max (loopDepth t) (loopDepth f))
  | .sum n b => max (loopDepth n) (1 + loopDepth b)

def Interface.nodes (f : Interface) : Nat :=
  (f.terms.map fun t => Pcv.nodes t.2).sum

def Interface.annotationCost (f : Interface) (s : Env) : Nat :=
  (f.terms.map fun t => cost t.2 s).sum

/-- Optional policy, separate from correctness. Neither budget is relative
    to program syntax size: a larger annotation can be much cheaper to run. -/
def Budgeted (p : Expr) (f : Interface) (maxNodes maxEval : Nat) : Prop :=
  Explains p f ∧ f.nodes ≤ maxNodes ∧ ∀ s, f.annotationCost s ≤ maxEval

/-- The tautological escape hatch, expressed in the SAME language.
    This transformation builds ordinary syntax; it is not an Expr primitive. -/
def costProgram : Expr → Expr
  | .lit _ | .var _ => .lit 0
  | .add a b | .mul a b | .sub a b | .div a b => .add (.add (costProgram a) (costProgram b)) (.lit 1)
  | .seq a b => .add (costProgram a) (costProgram b)
  | .iteLt a b t f => .add
      (.add (.add (costProgram a) (costProgram b)) (.lit 1))
      (.iteLt a b (costProgram t) (costProgram f))
  | .sum n b => .add (.add (costProgram n) (.lit 1))
      (.sum n (.add (costProgram b) (.lit 2)))

theorem costProgram_correct (p : Expr) (s : Env) :
    value (costProgram p) s = cost p s := by
  induction p generalizing s <;> simp_all [costProgram, value, cost]

/-- Preserve saturating natural subtraction when moving into the affine
    rational formula; ordinary field subtraction would be wrong at x < y. -/
theorem ratCast_sub (n m : Nat) :
    ((n - m : Nat) : Rat) = if m ≤ n then (n : Rat) - (m : Rat) else 0 := by
  by_cases h : m ≤ n
  · have he := congrArg (fun a : Nat => (a : Rat)) (Nat.sub_add_cancel h)
    simp at he
    simp [h]
    grind
  · have hn : n ≤ m := by omega
    simp [h, Nat.sub_eq_zero_of_le hn]

/-- Closing this tactic produces a kernel-checked proof of Explains.
    It is deliberately incomplete: failure means "not proved", not "false". -/
macro "perf_check" : tactic => `(tactic| (
  unfold Explains
  intro s
  simp [Interface.eval, cost, value, push,
    costProgram_correct, sumRange_index, sumRange_threshold, Nat.min_def, ratCast_sub]
  <;> (repeat' split)
  <;> (try simp_all)
  <;> grind))

macro "perf_check_budget" : tactic => `(tactic| (
  unfold Budgeted
  constructor
  · perf_check
  · constructor
    · decide
    · intro s
      simp [Interface.annotationCost, cost, value, push, sumRange_index,
        sumRange_threshold, Nat.min_def]
      <;> grind))

end Pcv
