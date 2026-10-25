/-
  Equality proofs between the C communication value models and GX's shallow
  PTX-side internal models. Each comment states the runtime or ISA behavior
  deliberately excluded from the equality.
-/
import Gxptx.Semantics.C.Communication

namespace Gxptx.CComm

/- PTX definition: each legal scalar atomic operation returns the old value and determines the corresponding replacement value under a type/operation table. -/
/- Lean modeling: case analysis proves equality only for the five represented fixed-width old/next functions; it proves no type legality, float/vector/b128/other-op behavior, atomicity, memory order, scope, address space, or runtime conformance. -/
theorem atomic_equiv (op : Ptx.AtomicOp) (old compare value : BitVec w) :
    atomic op old compare value = Ptx.atomic op old compare value := by
  cases op <;> rfl

/- PTX definition: `elect.sync` permits any deterministic participating-lane choice and sets true only for the elected lane. -/
/- Lean modeling: this definitional equality verifies GX's permitted least-lane choice against the same internal choice; least lane is not mandated by the PDF and rendezvous is not proved. -/
theorem elect_equiv (mask : Word) (lane : Nat) :
    elect mask lane = Ptx.elect mask lane := rfl

/- PTX definition: `shfl.sync` has mode-specific in-range rules, with butterfly requiring only the specified upper-bound check before its fallback. -/
/- Lean modeling: the corrected C model uses the same mode-specific range rule, so source selection agrees for every totalized input; membermask undefinedness remains outside this equality. -/
theorem shflSource_equiv (mode : Ptx.ShflMode)
    (lane selector clamp segmask : BitVec 5) (membermask : Word) :
    shflSource mode lane selector clamp segmask membermask =
      Ptx.shflSource mode lane selector clamp segmask membermask := by
  cases mode <;> rfl

/- PTX definition: an in-range shuffle reads the selected lane and otherwise returns the executing lane's own value with predicate false. -/
/- Lean modeling: rewriting the source equality proves unconditional agreement of the totalized value functions; membermask undefinedness and warp rendezvous remain unverified. -/
theorem shfl_equiv (mode : Ptx.ShflMode) (values : Nat → Word) (self : Word)
    (lane selector clamp segmask : BitVec 5) (membermask : Word) :
    shfl mode values self lane selector clamp segmask membermask =
      Ptx.shfl mode values self lane selector clamp segmask membermask := by
  rw [shfl, Ptx.shfl, shflSource_equiv]
  rfl

/- PTX definition: for butterfly mode, lane 16 xor selector 16 selects lane 0 and is in range because `0 ≤ maxLane = 31`. -/
/- Lean modeling: this executable regression proves both corrected models accept the cross-segment source when lanes 16 and 0 participate. -/
theorem shflBfly_crossSegment_regression :
    shflSource .bfly 16#5 16#5 31#5 16#5 0x00010001#32 = some 0 ∧
      Ptx.shflSource .bfly 16#5 16#5 31#5 16#5 0x00010001#32 = some 0 := by
  decide

/- PTX definition: upward shuffle cannot wrap below lane zero or below the packed lower clamp. -/
/- Lean modeling: executable cases check underflow, a segment boundary, and the first valid source above it. -/
theorem shflUp_boundary_regression :
    shflSource .up 0#5 1#5 0#5 0#5 0xffffffff#32 = none ∧
    shflSource .up 16#5 1#5 0#5 16#5 0xffffffff#32 = none ∧
    shflSource .up 17#5 1#5 0#5 16#5 0xffffffff#32 = some 16 ∧
    shflSource .up 8#5 1#5 8#5 0#5 0xffffffff#32 = none := by
  decide

/- PTX definition: a valid, unmasked `cp.async.bulk` makes its copied source bytes observable after completion. -/
/- Lean modeling: this definitional equality assumes legal operands and excludes masks, out-of-bounds modes, asynchronous execution, groups, signaling, visibility, ordering, overlap, and spaces. -/
theorem copyBytes_equiv (mem : Nat → UInt8) (dst src count : Nat) :
    copyBytes mem dst src count = Ptx.copyBytes mem dst src count := rfl

/- PTX definition: down does not wrap past lane31 and respects segmented upper bounds. -/
/- Lean modeling: executable cases exercise overflow, segment boundary, clamp, and valid source. -/
theorem shflDown_boundary_regression :
    shflSource .down 31#5 1#5 31#5 0#5 0xffffffff#32 = none ∧
    shflSource .down 15#5 1#5 15#5 16#5 0xffffffff#32 = none ∧
    shflSource .down 14#5 1#5 15#5 16#5 0xffffffff#32 = some 15 ∧
    shflSource .down 7#5 1#5 7#5 0#5 0xffffffff#32 = none ∧
    shflSource .down 0#5 31#5 31#5 0#5 0xffffffff#32 = some 31 := by
  decide

end Gxptx.CComm
