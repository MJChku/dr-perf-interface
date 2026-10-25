/-
  Value-level semantics for the PTX communication operations used by GX.

  Scheduling, rendezvous, and the host memory model remain runtime contracts;
  these shallow functions describe the value selected or written after those
  contracts complete, preserving the former `Gxptx.Core` definitions.
-/
import Gxptx.Semantics.Types
import Gxptx.Semantics.PTX.Arithmetic
import Gxptx.Semantics.PTX.Comparison

namespace Gxptx.Ptx

/- PTX definition: scalar `atom` has a type-dependent table containing add/inc/dec/min/max, bitwise and/or/xor, cas, exch, and floating forms. -/
/- Lean modeling: this tag set covers only scalar fixed-width add/cas/exch/min/max old/next values; it carries neither PTX type legality nor float/vector/b128/other-operation semantics. -/
inductive AtomicOp | add | cas | exch | minU | maxU | minS | maxS
  deriving Repr, DecidableEq

/- PTX definition: `atom` returns the original value and atomically stores the operation result. -/
/- Lean modeling: pairing `old` with `next` exposes both observable components of that single atomic transition. -/
structure AtomicResult (w : Nat) where
  old : BitVec w
  next : BitVec w
  deriving Repr, DecidableEq

/- PTX definition: a legal scalar integer/bit-pattern form atomically returns `*a` and replaces it with its operation result. -/
/- Lean modeling: each match arm returns the unchanged old value and computes its fixed-width next value only; legal type/operation combinations and the atomic memory event require separate checks. -/
def atomic (op : AtomicOp) (old compare value : BitVec w) : AtomicResult w :=
  match op with
  | .add => ⟨old, old + value⟩
  | .cas => ⟨old, if old = compare then value else old⟩
  | .exch => ⟨old, value⟩
  | .minU => ⟨old, Ptx.minU old value⟩
  | .maxU => ⟨old, Ptx.maxU old value⟩
  | .minS => ⟨old, Ptx.minS old value⟩
  | .maxS => ⟨old, Ptx.maxS old value⟩

/- PTX definition: `elect.sync` deterministically chooses one active lane from `membermask`. -/
/- Lean modeling: GX fixes the allowed deterministic choice to the least set lane and uses zero only for the unobservable empty-mask case. -/
def electedLane (mask : Word) : Nat :=
  ((List.range 32).find? fun lane => mask.getLsbD lane).getD 0

/- PTX definition: `elect.sync` returns the selected lane and marks only that lane's predicate true. -/
/- Lean modeling: reusing `electedLane` makes every participant agree, and the equality test makes exactly the selected lane true. -/
def elect (mask : Word) (lane : Nat) : Nat × Bool :=
  let selected := electedLane mask
  (selected, if mask = 0 then false else decide (lane = selected))

/- PTX definition: `shfl.sync` defines indexed and butterfly source-lane modes among its mode variants. -/
/- Lean modeling: the constructors name the source-selection modes currently emitted and verified by GX. -/
inductive ShflMode | idx | bfly | down | up
  deriving Repr, DecidableEq

/- PTX definition: `shfl.sync.idx` checks the segment range, while `shfl.sync.bfly` checks only `j ≤ maxLane`; sourcing a lane outside `membermask` is undefined. -/
/- Lean modeling: the mode-specific lower-bound condition transcribes the PDF rule; the membermask test is retained as a defined-domain guard for this total shallow function. -/
def shflSource (mode : ShflMode) (lane selector clamp segmask : BitVec 5)
    (membermask : Word) : Option Nat :=
  let minLane := lane &&& segmask
  let maxLane := minLane ||| (clamp &&& ~~~segmask)
  let source := match mode with
    | .idx => minLane ||| (selector &&& ~~~segmask)
    | .bfly => lane ^^^ selector
    | .down => lane + selector
    | .up => lane - selector
  let n := source.toNat
  match mode with
  | .idx =>
      if minLane.toNat ≤ n ∧ n ≤ maxLane.toNat ∧ membermask.getLsbD n
      then some n else none
  | .bfly =>
      if n ≤ maxLane.toNat ∧ membermask.getLsbD n then some n else none
  | .down =>
      if lane.toNat + selector.toNat ≤ maxLane.toNat ∧ membermask.getLsbD n
      then some n else none
  | .up =>
      if selector.toNat ≤ lane.toNat ∧ maxLane.toNat ≤ n ∧ membermask.getLsbD n
      then some n else none

/- PTX definition: valid `shfl.sync` reads lane `j`; an out-of-range source copies the caller's own value and clears `p`. -/
/- Lean modeling: `some` and `none` encode the documented range result on the defined membermask domain; rendezvous and undefined source-lane cases remain outside this value function. -/
def shfl (mode : ShflMode) (values : Nat → Word) (self : Word)
    (lane selector clamp segmask : BitVec 5) (membermask : Word) : Word × Bool :=
  match shflSource mode lane selector clamp segmask membermask with
  | some source => (values source, true)
  | none => (self, false)

/- PTX definition: a valid, unmasked `cp.async.bulk` copies its source range to the destination after completion. -/
/- Lean modeling: GX assumes legal size/alignment/ranges and models only completed bytes; masks, out-of-bounds modes, launch, signaling, ordering, overlap, and address spaces are excluded. -/
def copyBytes (mem : Nat → UInt8) (dst src count : Nat) : Nat → UInt8 :=
  fun i => if dst ≤ i ∧ i < dst + count then mem (src + (i - dst)) else mem i

end Gxptx.Ptx
