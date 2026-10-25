/-
  Total value certificates for communication instruction templates.

  These obligations concern returned or committed values.  Atomic certificates
  cover only the represented scalar fixed-width old/next functions, not PTX's
  type/operation legality table or floating/vector/b128 forms. Runtime atomicity,
  warp rendezvous, memory ordering, and asynchronous progress remain separate
  trusted/structural assumptions and known gaps in the validation inventory.
-/
import Gxptx.Verification.Communication

namespace Gxptx

/- PTX definition: PTX does not define this proof-obligation table; it specifies shuffle, copy, atomic, and election results separately. -/
/- Lean modeling: this total proposition assigns each finite `CommTemplate` constructor the exact shallow C/PTX value equality that it must satisfy. -/
def CommTemplate.spec : CommTemplate → Prop
  /- PTX definition: `atom.add` returns the old value and replaces it with the fixed-width sum. -/
  /- Lean modeling: the proposition compares only GX's shared old/next values; atomicity, order, and scope remain outside it. -/
  | .atomAdd => ∀ {w : Nat} (old compare value : BitVec w),
      CComm.atomic .add old compare value = Ptx.atomic .add old compare value
  /- PTX definition: `atom.cas` conditionally replaces the old value when it equals the comparison operand. -/
  /- Lean modeling: the equality covers returned and replacement values only, not atomic execution or memory-order behavior. -/
  | .atomCas => ∀ {w : Nat} (old compare value : BitVec w),
      CComm.atomic .cas old compare value = Ptx.atomic .cas old compare value
  /- PTX definition: `atom.exch` returns the old value and installs the supplied replacement. -/
  /- Lean modeling: the proposition compares the pure value transition while deliberately excluding runtime atomicity. -/
  | .atomExch => ∀ {w : Nat} (old compare value : BitVec w),
      CComm.atomic .exch old compare value = Ptx.atomic .exch old compare value
  /- PTX definition: unsigned `atom.min` replaces the old value only with a smaller unsigned operand. -/
  /- Lean modeling: both sides use fixed-width unsigned order for the pure replacement-value decision. -/
  | .atomMinU => ∀ {w : Nat} (old compare value : BitVec w),
      CComm.atomic .minU old compare value = Ptx.atomic .minU old compare value
  /- PTX definition: unsigned `atom.max` replaces the old value only with a greater unsigned operand. -/
  /- Lean modeling: both sides use fixed-width unsigned order for the pure replacement-value decision. -/
  | .atomMaxU => ∀ {w : Nat} (old compare value : BitVec w),
      CComm.atomic .maxU old compare value = Ptx.atomic .maxU old compare value
  /- PTX definition: signed `atom.min` compares operands using their signed interpretation. -/
  /- Lean modeling: both sides use `BitVec.toInt` for the pure signed replacement-value decision. -/
  | .atomMinS => ∀ {w : Nat} (old compare value : BitVec w),
      CComm.atomic .minS old compare value = Ptx.atomic .minS old compare value
  /- PTX definition: signed `atom.max` compares operands using their signed interpretation. -/
  /- Lean modeling: both sides use `BitVec.toInt` for the pure signed replacement-value decision. -/
  | .atomMaxS => ∀ {w : Nat} (old compare value : BitVec w),
      CComm.atomic .maxS old compare value = Ptx.atomic .maxS old compare value
  /- PTX definition: `elect.sync` returns true for one participating lane selected deterministically. -/
  /- Lean modeling: this obligation checks the permitted shared least-lane choice, not warp rendezvous or a uniquely mandated lane. -/
  | .elect => ∀ (mask : Word) (lane : Nat),
      CComm.elect mask lane = Ptx.elect mask lane
  /- PTX definition: `shfl.sync.idx` selects an indexed source lane subject to bounds and membership. -/
  /- Lean modeling: the corrected indexed value function agrees with the PTX-side model; warp rendezvous and undefined cases remain excluded. -/
  | .shflIdx => ∀ (values : Nat → Word) (self : Word)
      (lane selector clamp segmask : BitVec 5) (membermask : Word),
      CComm.shfl .idx values self lane selector clamp segmask membermask =
        Ptx.shfl .idx values self lane selector clamp segmask membermask
  /- PTX definition: `shfl.sync.bfly` selects an XOR-derived source subject to its range rule. -/
  /- Lean modeling: the corrected butterfly value function omits the invalid lower-bound test and agrees unconditionally; the cross-segment case is executable in `shflBfly_crossSegment_regression`. -/
  | .shflBfly => ∀ (values : Nat → Word) (self : Word)
      (lane selector clamp segmask : BitVec 5) (membermask : Word),
      CComm.shfl .bfly values self lane selector clamp segmask membermask =
        Ptx.shfl .bfly values self lane selector clamp segmask membermask
  /- PTX definition: down uses the upper clamp and a non-wrapping lane sum. -/
  /- Lean modeling: verifies the shallow value policy, not runtime rendezvous. -/
  | .shflDown => ∀ (values : Nat → Word) (self : Word)
      (lane selector clamp segmask : BitVec 5) (membermask : Word),
      CComm.shfl .down values self lane selector clamp segmask membermask =
        Ptx.shfl .down values self lane selector clamp segmask membermask
  /- PTX definition: `shfl.sync.up` uses non-wrapping subtraction with a lower clamp. -/
  /- Lean modeling: the pure result equality includes underflow rejection; it does not prove warp scheduling. -/
  | .shflUp => ∀ (values : Nat → Word) (self : Word)
      (lane selector clamp segmask : BitVec 5) (membermask : Word),
      CComm.shfl .up values self lane selector clamp segmask membermask =
        Ptx.shfl .up values self lane selector clamp segmask membermask
  /- PTX definition: a valid, unmasked `cp.async.bulk` exposes copied bytes in the destination after completion. -/
  /- Lean modeling: the equality assumes legal operands and excludes masks, out-of-bounds modes, progress, ordering, barriers, and address spaces. -/
  | .copyBytes => ∀ (mem : Nat → UInt8) (dst src count : Nat),
      CComm.copyBytes mem dst src count = Ptx.copyBytes mem dst src count

/- PTX definition: each modeled communication value result must agree with its instruction rule. -/
/- Lean modeling: exhaustive constructor matching supplies the corresponding equality theorem and rejects any newly added communication shape without a proof. -/
theorem CommTemplate.certified : ∀ k : CommTemplate, k.spec
  /- PTX definition: `atom.add` returns the old value and computes a wrapping-sum replacement. -/
  /- Lean modeling: specializing `atomic_equiv` to `.add` produces exactly the `.atomAdd` old/next value equality. -/
  | .atomAdd => CComm.atomic_equiv .add
  /- PTX definition: `atom.cas` conditionally replaces a matching old value. -/
  /- Lean modeling: specializing `atomic_equiv` to `.cas` produces exactly the `.atomCas` pure value equality. -/
  | .atomCas => CComm.atomic_equiv .cas
  /- PTX definition: `atom.exch` returns the old value and installs the supplied one. -/
  /- Lean modeling: specializing `atomic_equiv` to `.exch` produces exactly the `.atomExch` pure value equality. -/
  | .atomExch => CComm.atomic_equiv .exch
  /- PTX definition: unsigned `atom.min` uses unsigned operand order. -/
  /- Lean modeling: specializing `atomic_equiv` to `.minU` supplies the exact old/next value obligation. -/
  | .atomMinU => CComm.atomic_equiv .minU
  /- PTX definition: unsigned `atom.max` uses unsigned operand order. -/
  /- Lean modeling: specializing `atomic_equiv` to `.maxU` supplies the exact old/next value obligation. -/
  | .atomMaxU => CComm.atomic_equiv .maxU
  /- PTX definition: signed `atom.min` uses signed operand order. -/
  /- Lean modeling: specializing `atomic_equiv` to `.minS` supplies the exact old/next value obligation. -/
  | .atomMinS => CComm.atomic_equiv .minS
  /- PTX definition: signed `atom.max` uses signed operand order. -/
  /- Lean modeling: specializing `atomic_equiv` to `.maxS` supplies the exact old/next value obligation. -/
  | .atomMaxS => CComm.atomic_equiv .maxS
  /- PTX definition: `elect.sync` marks one deterministically selected participating lane. -/
  /- Lean modeling: `elect_equiv` directly proves the shared permitted lane-choice value required by `.elect.spec`. -/
  | .elect => CComm.elect_equiv
  /- PTX definition: `shfl.sync.idx` uses indexed source-lane selection. -/
  /- Lean modeling: the corrected mode-specific C model lets `shfl_equiv` supply unconditional result/predicate equality. -/
  | .shflIdx => CComm.shfl_equiv .idx
  /- PTX definition: `shfl.sync.bfly` uses XOR-derived source-lane selection. -/
  /- Lean modeling: the corrected helper policy lets `shfl_equiv` certify the cross-segment butterfly cases as well. -/
  | .shflBfly => CComm.shfl_equiv .bfly
  | .shflDown => CComm.shfl_equiv .down
  /- PTX definition: `shfl.sync.up` rejects subtraction below its clamp. -/
  /- Lean modeling: the shared range rule gives unconditional pure result/predicate equality. -/
  | .shflUp => CComm.shfl_equiv .up
  /- PTX definition: a valid, unmasked `cp.async.bulk` exposes copied destination bytes after completion. -/
  /- Lean modeling: `copyBytes_equiv` proves only the legal-operand post-completion byte map, not masks, out-of-bounds modes, or runtime properties. -/
  | .copyBytes => CComm.copyBytes_equiv

/- PTX definition: PTX does not define this project inventory; it separately defines the eleven represented communication forms. -/
/- Lean modeling: explicit enumeration gives the runtime checker a stable complete report list while the dependent certificate enforces proof totality. -/
def CommTemplate.all : List CommTemplate :=
  [.atomAdd, .atomCas, .atomExch, .atomMinU, .atomMaxU, .atomMinS, .atomMaxS,
   .elect, .shflIdx, .shflBfly, .shflDown, .shflUp, .copyBytes]

/- PTX definition: PTX does not define theorem-name metadata; this project table indexes proofs for the represented instruction families. -/
/- Lean modeling: this total mapping gives `gxcheck` the theorem name to resolve and audit for every enumerated communication template. -/
def CommTemplate.thmName : CommTemplate → String
  | .atomAdd | .atomCas | .atomExch | .atomMinU | .atomMaxU | .atomMinS | .atomMaxS =>
      "Gxptx.CComm.atomic_equiv"
  | .elect => "Gxptx.CComm.elect_equiv"
  | .shflIdx => "Gxptx.CComm.shfl_equiv"
  | .shflBfly => "Gxptx.CComm.shfl_equiv"
  | .shflDown => "Gxptx.CComm.shfl_equiv"
  | .shflUp => "Gxptx.CComm.shfl_equiv"
  | .copyBytes => "Gxptx.CComm.copyBytes_equiv"

end Gxptx
