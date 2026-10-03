# A same-language calculus for affine performance interfaces

The executable first prototype is in
[`experiments/lean-pcv`](../experiments/lean-pcv/README.md). It has been built and
checked in Lean 4.29.1. This document supersedes the earlier proposal that used
separate program and annotation languages.

## Objects

There is one language `Expr`, containing arithmetic, conditionals, sequencing,
and bounded loops. Both a program `P` and each PCV `phi_i` are terms of that
language. The program and PCVs are evaluated on the same entry environment.

```text
value : Expr -> Env -> Nat
cost  : Expr -> Env -> Nat
```

`value` describes the returned value; `cost` counts primitive operations under
an explicit unit-cost semantics. A loop executes its body repeatedly, with
an index bound in the environment. All terms terminate because loops are
bounded and the language contains no general recursion.

The only expression restriction on the resulting formula is affineness:

```text
cost(P, s) = d + a_1*value(phi_1, s) + ... + a_k*value(phi_k, s)
```

The coefficients are fixed rational numbers. PCVs may themselves branch,
loop, multiply, or divide. The first prototype proves identities for total
cost; per-source-location counts would be a further extension matching
DynamoRIO drperf's blockwise analysis.

## Automatic discovery and checking

The user supplies P and the PCVs, not coefficients or loop invariants:

```lean
perf_discover thresholdAuto : threshold using [x, expensive]
```

The implementation executes a small input grid and solves exact rational
linear equations to propose coefficients. It then generates an all-input
proposition and invokes the `perf_check` tactic. Successful compilation means
Lean has checked the resulting proof.

The generic rules compose sequence costs, select branch costs, and sum loop
costs. Reusable, inductively proved identities summarize constant sums,
products, arithmetic progressions, and threshold branches. Examples supply
no additional invariants or handwritten derivations. This is an initial
library of automatic steps, not a complete decision procedure for the language.

The trust distinction is:

```text
coefficient proposal: executions + exact linear algebra
acceptance:           kernel-checked proof for every input
```

A regression test fits a formula on every discovery sample but changes
behavior at x=1000. The final proof rejects that proposal.

## What has been proved

For the threshold loop, the checker automatically discovers and proves:

```text
E = if x < y then x else y
cost = 4*x + 2*E + 1
```

A separate theorem proves that no rational coefficients in raw x,y can
explain the same program on all inputs.

For the triangular nested loop, it discovers and proves:

```text
pairs = x*(x-1)/2
cost = 3*pairs + 3*x + 1
```

A PCV computing pairs by a loop is also accepted. Both use the same language;
they differ in complexity, not eligibility.

## Why correctness alone is insufficient

A transformation `costProgram(P)` builds an ordinary term of the same language
whose value equals P's cost. Lean proves this for every program and input:

```text
value(costProgram(P), s) = cost(P, s)
```

Consequently a one-PCV, coefficient-one exact explanation always exists in
this total language. There is no special cost-query primitive: the transformed
term recomputes the necessary work using ordinary arithmetic, branches, and
loops.

The useful question is therefore whether an explanation is compact and cheap.
The prototype separately checks:

```text
Explains(P, interface)
PCV syntax nodes <= size budget
for every input: PCV evaluation cost <= evaluation budget
```

This already exposes a tradeoff. The triangular source has 7 syntax nodes;
its closed-form PCVs have 8 nodes and cost 3 operations to evaluate. A smaller
4-node annotation uses a loop and costs 2*x+1 operations. For x=100 the original
program costs 15,151 operations. Syntax size alone is not a sufficient objective.

The next mathematical questions are about tradeoffs among feature count,
annotation size, annotation evaluation cost, and proof effort. Universal
expressibility by itself is already settled for total cost in this toy language.

## Boundaries

This is a formal experiment alongside drperf, not a replacement for the
DynamoRIO checker. Its guarantees concern its stated source-operation model,
with arbitrary natural-number inputs, no mutation, and bounded loops. They
are not proofs of machine instruction formulas.

The inference grid is finite; its role is proposal generation only. The proof
procedure is incomplete, so failure may mean either a false proposal or a
missing automatic proof step. The expected-failure tests and a separate
non-expressibility theorem distinguish those situations where possible.

Run `python3 experiments/lean-pcv/validate.py` from the repository root to
build the proofs, audit their axioms, test rejection, and print the demo.
