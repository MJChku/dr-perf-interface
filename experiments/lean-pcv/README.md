# Automatic affine interfaces in Lean

The separate [waited algebra](../../docs/waited-algebra.md) in `Waited.lean`
formalizes nested region dependencies and safe simplification. Run
`lake build WaitedExamples` for its checked examples; it does not change the
affine cost language below.

The input is an inlined program and a list of PCVs. Programs and PCVs have
exactly the same language, including loops. The checker infers coefficients
and automatically proves the resulting cost identity for all natural-number
inputs. No example-specific invariants or proof steps are supplied to the
discovery command.

```sh
cd experiments/lean-pcv
python3 validate.py
```

Requires Lean 4.29.1 and Python 3. No Mathlib or downloaded Lake dependencies.
For just the proofs and readable demo, run `lake build` and `lake exe pcv-demo`.

## User-facing examples

The primary examples are in `Inferred.lean`:

```lean
perf_discover linearAuto : linear using [x]
perf_discover rectangularAuto : rectangular using [.mul x y, x]
perf_discover thresholdAuto : threshold using [x, expensive]
perf_discover triangularAuto : triangular using [pairs, x]
```

Here `expensive` is `if x < y then x else y`, and `pairs` is
`x*(x-1)/2`, both represented in the same `Expr` language as the programs.
`perf_discover` generates an interface definition and an accompanying
`<name>_checked` theorem. The local simplifier declarations only unfold syntax
definitions; they supply no program-specific cost lemmas or invariants.

The programs really contain loops. `Expr.sum bound body` executes a counted
loop and accumulates its body's values. For example, `triangular` is:

```text
result = 0
for i = 0 .. x-1:
    inner = 0
    for j = 0 .. i-1:
        inner += (0 + 0)
    result += inner
```

`threshold` is:

```text
result = 0
for i = 0 .. x-1:
    if i < y:
        result += ((0 + 0) + (0 + 0))
    else:
        result += (0 + 0)
```

These are abstract operations: there is no compiler optimizing away the
additions. The syntax is evaluated directly under the specified cost model.

## Results

All of these coefficients are inferred and then proved:

```text
linear:                 3*x + 1
rectangular:            3*(x*y) + 3*x + 1
threshold:              4*x + 2*min(x,y) + 1
threshold, branch PCVs:  6*min(x,y) + 4*(x-min(x,y)) + 1
threshold, signed:      6*x - 2*(x-min(x,y)) + 1
triangular:             3*(x*(x-1)/2) + 3*x + 1
triangular, loop PCV:    3*sum(i<x,i) + 3*x + 1
fractional:             (3/2)*(2*x) + 1
recompute cost:         1*costProgram(P) + 0
```

The triangular loop-PCV example uses a real loop as a PCV. It reduces a
quadratic computation to a linear annotation; the closed-form version reduces
annotation evaluation to three operations.

`Examples.threshold_not_raw` additionally proves that **no rational affine
coefficients** in raw `x,y` can explain the threshold program. This is stronger
than a tactic failing to find a proof.

## How the automatic checker works

1. Execute the program and PCVs on a small grid of inputs.
2. Solve for coefficients by exact rational Gaussian elimination.
3. Generate the proposition `forall s, cost P s = affine(PCVs(s))`.
4. Run `perf_check`: reduce the compositional cost semantics, apply generic
   loop-sum lemmas, split remaining guards, and solve the arithmetic.
5. Lean's kernel checks the resulting proof.

The samples propose a formula; they never establish acceptance. The native
coefficient-discovery code is outside the trusted proof path. Generic lemmas
cover constant sums, additive composition, factoring products, arithmetic
progressions, and threshold branches. Their inductive proofs live in the
checker library and are reused by every example.

This does perform compositional symbolic reasoning. It does not unroll a
symbolic number of iterations in the supported loop families. The affine
shape does not itself solve every possible loop-sum identity.

`tests/RejectUnseen.lean` deliberately changes behavior at `x=1000`, beyond
all discovery samples. Coefficient fitting succeeds, but the universal proof
fails. `validate.py` checks this expected failure separately from rejection of
inconsistent samples in `tests/RejectRaw.lean`.

## Correctness and complexity are separate checks

`costProgram` mechanically builds an ordinary `Expr` whose returned value
is the original program's cost. It introduces no cost-query primitive into
the language. `costProgram_correct` proves this for every program and input,
and `cheating_always_checks` proves that the coefficient-one interface works.

That makes a complexity criterion essential. The prototype separately records
total PCV syntax nodes and the cost of evaluating the PCVs. `Budgeted` and
`perf_check_budget` can prove bounds on both, in addition to the cost identity.
The budget does not require the annotation to be syntactically smaller than P.

```text
Threshold at x=100, y=30:
                         nodes   evaluation operations
  original program          15                    461
  compact PCVs               6                      1
  recompute-cost PCV        35                   1023

Triangular loop at x=100:
  original program           7                  15151
  closed-form PCVs           8                      3
  loop PCVs                  4                    201
  recompute-cost PCV        21                  25353
```

The closed form has more syntax than the original triangular program but is
much cheaper to evaluate. Minimizing syntax alone would prefer the slower
loop annotation. Constant evaluation budgets of 1 and 3 are proved for the
compact threshold and triangular annotations on **all** inputs. The generated
threshold recomputation fails these budgets; a separate theorem proves its
evaluation exceeds the runtime budget even without a syntax limit.

PCV evaluation costs exclude the final affine combination. Syntax counts are
AST node counts, with no sharing and with each literal counting as one node.

## Exact scope

- One total language: natural literals, variable reads, arithmetic, sequencing,
  conditionals, and bounded summation loops. No calls, general recursion,
  mutable state, heap, or early exits. Lean definitions naming syntax fragments
  are abbreviations, not function calls in the object language.
- Natural subtraction saturates at zero. Division follows Lean's natural
  division, including a zero result for division by zero.
- Arithmetic and comparisons cost one each; reads and literals cost zero.
  A loop evaluates its bound once, charges one final test, and charges a test
  and an accumulation per iteration, plus the body cost. Arithmetic is unit
  cost even for arbitrarily large naturals; this is not a bit-complexity model.
- Coefficients are signed rationals. The identity concerns total source cost,
  not DynamoRIO instruction counts or the repo's per-basic-block fitting.
- Discovery currently uses `{0,1,2,3,4,7,11}` for up to three free input slots.
  Rank deficiency is reported as insufficient variation, not non-expressibility.
- The automatic proof procedure is sound when it succeeds, but incomplete.
  A failed proof means unproved; it does not by itself mean the PCVs are wrong.
- Correctness and compression are independent properties. A verified interface
  need not be efficient unless a complexity budget is checked too.

## Files

- `Pcv.lean`: language, semantics, loop lemmas, cost reification, and tactics.
- `Discover.lean`: coefficient proposal and the `perf_discover` command.
- `Examples.lean`: source programs and foundational positive/negative proofs.
- `Inferred.lean`: the user-facing examples with no supplied coefficients.
- `Audit.lean`: axiom inspection of fourteen key theorems.
- `validate.py`: builds proofs/demo, audits axioms, and checks rejection cases.

The successful audited theorems use only Lean's standard `propext`,
`Classical.choice`, and `Quot.sound` axioms. No `sorry`, user axioms, or native
evaluation proof shortcuts are needed. Expected-failure files are deliberately
excluded from the successful build.
