# Waited relations: a checked algebra

The implementation is [Waited.lean](../experiments/lean-pcv/Waited.lean), with
[positive and negative examples](../experiments/lean-pcv/WaitedExamples.lean).
It uses Lean 4.29.1, without Mathlib or additional dependencies:

```sh
cd /home/ubuntu/drperf/experiments/lean-pcv
lake build WaitedExamples
```

## Objects and symbols

- `R_A`, `R_B`: region **invocations**, with separate identities even when their
  source names are equal. A region has a set of represented event occurrences.
- `e`: a particular publication occurrence, including its generation.
- `w`: the event occurrence at the existing `waited` checkpoint. This is not an
  additional wait-begin annotation.
- `R_B ⊆ R_A`: scope inclusion: every represented event in B belongs to A.
  Scopes come from structural invocation membership, not wall-clock overlap.
- `R_C ⊢ₚ e`: publication e belongs to C.
- `e ≺ w`: the ordering constraint that e precedes w.
- `R_A ⊣[τ, w] e`: in timeline τ, w belongs to A and e precedes w.
- `Γ`: the constraints retained in the model; `Γ ⊨ p` means every timeline
  satisfying Γ also satisfies p.

Publication ownership and scope membership are inputs to this algebra. It does
not establish that an annotation was placed at the application's real release,
or determine an unknown native object's producer.

## Hiding a subregion versus deleting its constraint

Scope inclusion supports this projection:

```text
R_B ⊆ R_A     R_C ⊆ R_D
R_B ⊣[τ, w] e     R_C ⊢ₚ e
─────────────────────────────────
R_A ⊣[τ, w] e     R_D ⊢ₚ e
```

The checkpoint **w remains exactly the same**. A display can hide B and C and
show the dependency within A and D, retaining e and w as evidence. This does not
say that all of D finishes before A starts. It removes no synchronization from
the application and changes no instruction accounting.

Deleting an ordering assertion has a stronger requirement:

```text
Γ ⊨ p    iff    ∀ τ, (τ satisfies Γ ∪ {p}) ↔ (τ satisfies Γ)
```

This equivalence is proved as `drop_iff_entails`. `canDrop? Γ p` additionally
requires `Sat Γ`, so contradictory premises cannot justify simplification.
Γ contains the **other retained constraints**, not the assertion being tested.

## When the parent's waited covers the child's

Write w_A and w_B for their actual checkpoints. A sufficient rule is:

```text
R_B ⊆ R_A
Γ ⊨ e ≺ w_A      w_A ≼[Γ] w_B
─────────────────────────────────
Γ ⊨ e ≺ w_B
```

Here `x ≼[Γ] y` means x and y are the same event, or Γ entails x ≺ y. A parent's
earlier checkpoint can make the later child assertion redundant. A projected
parent assertion retaining the child's checkpoint is the equality case.

If the actual outer `waited` comes **after** B, the implication goes the other
way: the child's constraint can imply the parent's. Merely containing B and
executing a later `waited` does not cover B's earlier checkpoint. Counterexample:

```text
w_B ≺ e ≺ w_A
```

A's assertion is satisfied, while B's assertion e ≺ w_B is violated. The Lean
examples check that removing B here changes an inconsistent model into a
consistent one, and therefore reject the deletion.

Different releases can also be covered when an explicit dependency connects them:

```text
e_B ≼[Γ] e_A     Γ ⊨ e_A ≺ w_A     w_A ≼[Γ] w_B
────────────────────────────────────────────────
Γ ⊨ e_B ≺ w_B
```

This is `covers_via_order`. No publisher equality or dependency is invented.

## Shared helpers, composition, and unknown publications

Different invocations B₁ and B₂ retain different scopes and event identities.
Matching their source name `sync` cannot connect their callers. Compose concrete
relations before aggregating names for display.

For transitivity through B, its incoming checkpoint must precede its outgoing
publication: e_C ≺ w_B ≺ e_B ≺ w_A. If B publishes to A before waiting for C,
the region arrows alone cannot establish that A depends on C.

For `Wait[?]`, a convenient satisfying assignment of the unknown publisher is not
coverage. `coversUnknown?` conservatively requires coverage of **every** supplied
candidate publication. Using it requires that the actual publisher belongs to
that candidate set; the algebra does not establish the set's completeness.
Even when every candidate is ordered early enough, this proves order coverage,
not the identity of the native producer. An empty candidate set is rejected.

## What is checked

- `accepts? τ Γ`: does the **recorded** timeline respect the constraints?
- `sat? Γ`: does **any** timeline satisfy them?
- `entails? Γ p`: does every satisfying timeline already enforce p?
- `canDrop? Γ p`: is Γ consistent and does it entail p?

The Lean proofs establish soundness and completeness of `sat?` and `entails?`
for finite sets of strict event-order constraints over natural timestamps.
Rank compression is proved to preserve every strict comparison; therefore
enumerating ranks 0 through n−1 covers arbitrary timestamps for n events.
The reference algorithm enumerates n^n assignments and is intended for small
examples, not production profiles. A production implementation should use a
proved cycle/reachability algorithm or checked certificates.

The positive/negative examples cover: lifting, a later parent's insufficient
wait, an earlier parent's coverage, equal checkpoints, compatible but
nonredundant claims, shared helpers, unknown publishers, premature publication,
connected dependency chains, wrong recorded order, cycles, and vacuous entailment.
The audited proofs use only standard Lean axioms; no `sorry`, custom axioms, or
`native_decide` shortcuts are used.

For drperf integration, Γ must contain justified execution-order and dependency
constraints. Adding every accidental cross-thread timestamp ordering would
make observed waits trivially redundant and defeat the check. This prototype
does not yet import drperf reports or change their graph simplification. It
formalizes when that simplification is valid; it does not prove a program's
declarations correct on unobserved executions.
