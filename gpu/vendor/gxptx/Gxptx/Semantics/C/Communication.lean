/-
  Shallow value models of the C/runtime helpers used for the communication
  instructions currently handled by GX. Runtime scheduling and memory-order
  contracts are intentionally outside these definitions.
-/
import Gxptx.Semantics.PTX.Communication

namespace Gxptx

/- PTX definition: PTX does not define this GX helper-tag inventory; it specifies the represented communication operations separately. -/
/- Lean modeling: this tag type records exactly the atomic, election, restricted shuffle, and completed-copy helper shapes currently printed by GX. -/
inductive CommTemplate
  /- PTX definition: `atom.add` returns the old value and atomically stores `old + value`. -/
  /- Lean modeling: this inventory tag identifies a value helper only; atomicity, scope, ordering, and runtime execution are outside it. -/
  | atomAdd
  /- PTX definition: `atom.cas` conditionally exchanges memory when its old value equals compare. -/
  /- Lean modeling: this inventory tag identifies the old/next value helper but carries no indivisibility or memory-order semantics. -/
  | atomCas
  /- PTX definition: `atom.exch` returns the old value and atomically stores the supplied value. -/
  /- Lean modeling: this inventory tag identifies the exchange value helper while leaving the runtime atomic transition unmodeled. -/
  | atomExch
  /- PTX definition: unsigned `atom.min` stores the lesser unsigned value and returns the old value. -/
  /- Lean modeling: the tag selects unsigned value ordering but does not certify helper atomicity, scope, or ordering. -/
  | atomMinU
  /- PTX definition: unsigned `atom.max` stores the greater unsigned value and returns the old value. -/
  /- Lean modeling: the tag selects unsigned value ordering but does not certify helper atomicity, scope, or ordering. -/
  | atomMaxU
  /- PTX definition: signed `atom.min` stores the lesser signed value and returns the old value. -/
  /- Lean modeling: the tag selects signed value ordering but does not certify helper atomicity, scope, or ordering. -/
  | atomMinS
  /- PTX definition: signed `atom.max` stores the greater signed value and returns the old value. -/
  /- Lean modeling: the tag selects signed value ordering but does not certify helper atomicity, scope, or ordering. -/
  | atomMaxS
  /- PTX definition: `elect.sync` selects one deterministic lane from the participating set. -/
  /- Lean modeling: GX chooses the least active lane, which is permitted, but this tag does not model synchronization or participation. -/
  | elect
  /- PTX definition: `shfl.sync.idx` derives an indexed source lane within its segment. -/
  /- Lean modeling: the tag routes to GX's indexed source policy; membermask and undefined-participation contracts remain outside it. -/
  | shflIdx
  /- PTX definition: `shfl.sync.bfly` derives its source lane by xor with the selector. -/
  /- Lean modeling: this tag routes to GX's butterfly source policy; membermask and undefined-participation contracts remain outside it. -/
  | shflBfly
  /- PTX definition: downward shuffle adds the lane offset and checks the upper clamp without wrapping. -/
  /- Lean modeling: the runtime tag uses mode three; synchronization remains a runtime contract. -/
  | shflDown
  /- PTX definition: `shfl.sync.up` selects lane minus offset, bounded below by the packed clamp. -/
  /- Lean modeling: this tag preserves the non-wrapping subtraction range and result predicate; rendezvous remains a runtime contract. -/
  | shflUp
  /- PTX definition: a valid, unmasked `cp.async.bulk` transfers its requested source bytes to destination after completion. -/
  /- Lean modeling: the tag assumes legal size/alignment/ranges and excludes masks, out-of-bounds modes, launch, barriers, visibility, ordering, overlap, and address spaces. -/
  | copyBytes
  deriving Repr, DecidableEq

namespace CComm

/- PTX definition: `atom` returns the old value and stores the add/cas/exch/min/max result as one atomic transition. -/
/- Lean modeling: this preserves only GX's old/next value transcription; indivisibility, scope, ordering, address-space rules, and runtime execution are not modeled here. -/
def atomic (op : Ptx.AtomicOp) (old compare value : BitVec w) : Ptx.AtomicResult w :=
  match op with
  | .add => ⟨old, old + value⟩
  | .cas => ⟨old, if old = compare then value else old⟩
  | .exch => ⟨old, value⟩
  | .minU => ⟨old, Ptx.minU old value⟩
  | .maxU => ⟨old, Ptx.maxU old value⟩
  | .minS => ⟨old, Ptx.minS old value⟩
  | .maxS => ⟨old, Ptx.maxS old value⟩

/- PTX definition: `elect.sync` deterministically chooses one active lane, without requiring that it be the least-numbered lane. -/
/- Lean modeling: GX reuses its permitted least-set-lane choice and returns true only for that lane; this models the chosen value policy, not rendezvous. -/
def elect (mask : Word) (lane : Nat) : Nat × Bool :=
  let selected := Ptx.electedLane mask
  (selected, if mask = 0 then false else decide (lane = selected))

/- PTX definition: `shfl.sync.{idx,bfly}` computes a source lane from selector, clamp, segment mask, and mode. -/
/- Lean modeling: the mode-specific range tests transcribe the indexed and butterfly source rules; the membermask test totalizes an ISA-undefined source-participation case. -/
def shflSource (mode : Ptx.ShflMode) (lane selector clamp segmask : BitVec 5)
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

/- PTX definition: a valid `shfl.sync` source supplies its value, while an out-of-range source copies the caller's value and clears the optional predicate. -/
/- Lean modeling: this preserves GX's `Option`-based source policy exactly; membermask undefinedness and warp rendezvous remain outside the value function. -/
def shfl (mode : Ptx.ShflMode) (values : Nat → Word) (self : Word)
    (lane selector clamp segmask : BitVec 5) (membermask : Word) : Word × Bool :=
  match shflSource mode lane selector clamp segmask membermask with
  | some source => (values source, true)
  | none => (self, false)

/- PTX definition: a valid, unmasked `cp.async.bulk` has copied its requested source range after completion. -/
/- Lean modeling: GX assumes legal size/alignment/ranges and models only completed bytes; masks, out-of-bounds modes, launch, groups, visibility, ordering, overlap, and spaces are excluded. -/
def copyBytes (mem : Nat → UInt8) (dst src count : Nat) : Nat → UInt8 :=
  fun i => if dst ≤ i ∧ i < dst + count then mem (src + (i - dst)) else mem i

end CComm
end Gxptx
