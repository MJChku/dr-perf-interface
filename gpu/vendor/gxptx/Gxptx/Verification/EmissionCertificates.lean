/-
  Total certificates for proved emission families outside `CTemplate`.

  Conversion, move, widening/high multiplication, memory address/value, and
  bit-manipulation models are indexed here so adding an emission kind without
  a specification or proof is rejected by Lean's exhaustiveness checker.
-/
import Gxptx.Verification.Conversion
import Gxptx.Verification.Move
import Gxptx.Verification.Arithmetic
import Gxptx.Verification.Memory
import Gxptx.Verification.BitManip

namespace Gxptx

/- PTX definition: these are the currently proved non-`CTemplate` instruction shapes. -/
/- Lean modeling: one finite constructor names each separately certified C/PTX value model, enabling exhaustive specification and reporting. -/
inductive EmitKind
  /- PTX definition: integer `cvt` extends or truncates according to type widths and source signedness. -/
  /- Lean modeling: one constructor makes the quantified integer-conversion certificate independently enumerable and reviewable. -/
  | cvtII
  /- PTX definition: scalar `mov.bN` copies every source bit unchanged. -/
  /- Lean modeling: the separate constructor identifies the width-generic scalar identity obligation. -/
  | movB
  /- PTX definition: brace-source `mov` packs the first operand low and second high. -/
  /- Lean modeling: the separate constructor identifies the equal-half-width packing obligation. -/
  | movPack
  /- PTX definition: the first brace destination receives the packed source's low half. -/
  /- Lean modeling: the separate constructor identifies low extraction from an `h + h` source. -/
  | movUnpackLo
  /- PTX definition: the second brace destination receives the packed source's high half. -/
  /- Lean modeling: the separate constructor identifies high extraction from an `h + h` source. -/
  | movUnpackHi
  /- PTX definition: `mul.wide.uN` is legal only for N = 16 or 32 and returns the complete 2N-bit product. -/
  /- Lean modeling: the tag tracks a generic algebraic equality; it is a PTX certificate only at those legal specializations. -/
  | mulWideU
  /- PTX definition: `mul.wide.sN` is legal only for N = 16 or 32 and returns the complete 2N-bit product. -/
  /- Lean modeling: the tag tracks a generic algebraic equality; it is a PTX certificate only at those legal specializations. -/
  | mulWideS
  /- PTX definition: `mul.hi.uN` is legal only for N = 16, 32, or 64 and returns the upper N bits. -/
  /- Lean modeling: the tag tracks a generic algebraic equality; it is a PTX certificate only at those legal specializations. -/
  | mulHiU
  /- PTX definition: `mul.hi.sN` is legal only for N = 16, 32, or 64 and returns the upper N bits. -/
  /- Lean modeling: the tag tracks a generic algebraic equality; it is a PTX certificate only at those legal specializations. -/
  | mulHiS
  /- PTX definition: `mad.hi.uN` is legal only for N = 16, 32, or 64 and adds `c` to the high half. -/
  /- Lean modeling: the tag tracks a generic algebraic equality; it is a PTX certificate only at those legal specializations. -/
  | madHiU
  /- PTX definition: `mad.hi.sN` is legal only for N = 16, 32, or 64 and adds `c` to the high half. -/
  /- Lean modeling: the tag tracks a generic algebraic equality; it is a PTX certificate only at those legal specializations. -/
  | madHiS
  /- PTX definition: `mad.wide.uN` is legal only for N = 16 or 32 and adds a 2N-bit addend to the full product. -/
  /- Lean modeling: the tag tracks a generic equality whose explicit legal-width wrappers enforce the PTX operand/result shape. -/
  | madWideU
  /- PTX definition: `mad.wide.sN` is legal only for N = 16 or 32 and adds a 2N-bit addend to the full signed product. -/
  /- Lean modeling: the tag tracks a generic equality whose explicit legal-width wrappers enforce the PTX operand/result shape. -/
  | madWideS
  /- PTX definition: an element address is base plus index times element size. -/
  /- Lean modeling: the constructor separates address-value verification from load and store result policies. -/
  | ldStElemAddr
  /- PTX definition: `ld.global` reads the addressed value; fake-address suppression is GX policy. -/
  /- Lean modeling: the constructor tracks GX's abstract read-result policy without claiming concrete memory semantics. -/
  | gld
  /- PTX definition: `st.global` writes the addressed value; fake-address suppression is GX policy. -/
  /- Lean modeling: the constructor tracks GX's abstract memory-update policy without claiming concrete effects. -/
  | gst
  /- PTX definition: `shf.r.b32` right-shifts `b:a` and returns its low word. -/
  /- Lean modeling: the constructor tracks whole-result equality for every modeled amount up to 32. -/
  | shfR
  /- PTX definition: `shf.l.b32` left-shifts `b:a` and returns its high word. -/
  /- Lean modeling: the constructor tracks whole-result equality for every modeled amount up to 32. -/
  | shfL
  /- PTX definition: `.wrap` selects the amount from the low five control bits. -/
  /- Lean modeling: the constructor separately records equality of masking and modulo-32 amount selection. -/
  | shfAmtWrap
  /- PTX definition: `.clamp` limits the unsigned control amount to 32. -/
  /- Lean modeling: the constructor separately records equality of conditional and minimum-based amount selection. -/
  | shfAmtClamp
  /- PTX definition: `bfi` inserts field bits and preserves base bits elsewhere. -/
  /- Lean modeling: the constructor tracks the modeled helper's complete in-range per-bit obligation. -/
  | bfi
  /- PTX definition: `bfe.uN` extracts selected bits and zero-fills the rest. -/
  /- Lean modeling: the constructor explicitly tracks only unsigned extraction under the current helper assumptions. -/
  | bfeU
  /- PTX definition: generic `prmt.b32` assembles four selector-chosen bytes from `b:a`. -/
  /- Lean modeling: the constructor tracks generic-mode whole-result equality while specialized modes remain outside scope. -/
  | prmt
  deriving Repr, DecidableEq

/- PTX definition: each indexed emission kind must satisfy its instruction's value equation. -/
/- Lean modeling: the total dependent proposition assigns the exact PTX-vs-C obligation to every constructor and makes a missing case a compile-time error. -/
def EmitKind.spec : EmitKind → Prop
  /- PTX definition: integer `cvt` sign- or zero-extends on widening and truncates on narrowing. -/
  /- Lean modeling: quantifying source width, destination width, and signedness states the complete modeled integer-cast value obligation. -/
  | .cvtII => ∀ (sgn : Bool) (dw : Nat) {sw : Nat} (a : BitVec sw),
      CCvt.evalC sgn dw a = Ptx.cvtI sgn dw a
  /- PTX definition: scalar `mov.bN` copies every source bit unchanged. -/
  /- Lean modeling: the proposition compares the same width-generic source value in both shallow models. -/
  | .movB => ∀ {w : Nat} (a : BitVec w), CMov.movB a = Ptx.movB a
  /- PTX definition: brace-source `mov` packs the first element low and the second high. -/
  /- Lean modeling: quantifying half-width operands compares C-side shift/or packing with the same `hi ++ lo` value. -/
  | .movPack => ∀ {h : Nat} (lo hi : BitVec h), CMov.pack lo hi = Ptx.movPack lo hi
  /- PTX definition: the first brace destination receives the packed source's low half. -/
  /- Lean modeling: the `BitVec (h + h)` premise fixes a source containing exactly two equal-width components. -/
  | .movUnpackLo => ∀ {h : Nat} (a : BitVec (h + h)), CMov.unpackLo a = Ptx.movUnpackLo a
  /- PTX definition: the second brace destination receives the packed source's high half. -/
  /- Lean modeling: the `BitVec (h + h)` premise fixes a source containing exactly two equal-width components. -/
  | .movUnpackHi => ∀ {h : Nat} (a : BitVec (h + h)), CMov.unpackHi a = Ptx.movUnpackHi a
  /- PTX definition: `mul.wide.uN` is legal only for N = 16 or 32 and writes the full 2N-bit product. -/
  /- Lean modeling: this proposition is generic algebraically and denotes a PTX obligation only at those legal widths. -/
  | .mulWideU => ∀ {w : Nat} (a b : BitVec w), CW.mulWideU a b = Ptx.mulWideU a b
  /- PTX definition: `mul.wide.sN` is legal only for N = 16 or 32 and writes the full 2N-bit product. -/
  /- Lean modeling: this proposition is generic algebraically and denotes a PTX obligation only at those legal widths. -/
  | .mulWideS => ∀ {w : Nat} (a b : BitVec w), CW.mulWideS a b = Ptx.mulWideS a b
  /- PTX definition: `mul.hi.uN` is legal only for N = 16, 32, or 64 and returns the upper half. -/
  /- Lean modeling: this proposition is generic algebraically and denotes a PTX obligation only at those legal widths. -/
  | .mulHiU => ∀ {w : Nat} (a b : BitVec w), CW.mulHiU a b = Ptx.mulHiU a b
  /- PTX definition: `mul.hi.sN` is legal only for N = 16, 32, or 64 and returns the upper half. -/
  /- Lean modeling: this proposition is generic algebraically and denotes a PTX obligation only at those legal widths. -/
  | .mulHiS => ∀ {w : Nat} (a b : BitVec w), CW.mulHiS a b = Ptx.mulHiS a b
  /- PTX definition: `mad.hi.uN` is legal only for N = 16, 32, or 64. -/
  /- Lean modeling: this proposition is generic algebraically and denotes a PTX obligation only at those legal widths. -/
  | .madHiU => ∀ {w : Nat} (a b c : BitVec w), CW.madHiU a b c = Ptx.madHiU a b c
  /- PTX definition: `mad.hi.sN` is legal only for N = 16, 32, or 64. -/
  /- Lean modeling: this proposition is generic algebraically and denotes a PTX obligation only at those legal widths. -/
  | .madHiS => ∀ {w : Nat} (a b c : BitVec w), CW.madHiS a b c = Ptx.madHiS a b c
  /- PTX definition: `mad.wide.uN` adds a 2N-bit addend to the full unsigned product for legal N = 16 or 32. -/
  /- Lean modeling: the dependent addend/result width records the widening shape, while legal wrappers restrict PTX certification to 16/32-bit sources. -/
  | .madWideU => ∀ {w : Nat} (a b : BitVec w) (c : BitVec (2 * w)),
      CW.madWideU a b c = Ptx.madWideU a b c
  /- PTX definition: `mad.wide.sN` adds a 2N-bit addend to the full signed product for legal N = 16 or 32. -/
  /- Lean modeling: the dependent addend/result width records the widening shape, while legal wrappers restrict PTX certification to 16/32-bit sources. -/
  | .madWideS => ∀ {w : Nat} (a b : BitVec w) (c : BitVec (2 * w)),
      CW.madWideS a b c = Ptx.madWideS a b c
  /- PTX definition: element `i` is addressed at `base + i * sizeof(element)`. -/
  /- Lean modeling: the proposition compares the same 64-bit wrapping base and natural byte offset on both sides. -/
  | .ldStElemAddr => ∀ (base : BitVec 64) (i sz : Nat),
      CMem.elemAddr base i sz = Ptx.elemAddr base i sz
  /- PTX definition: `ld.global` reads the addressed value; GX's fake-address guard is runtime policy. -/
  /- Lean modeling: quantifying an abstract read function isolates equality of GX's guard/result policy from concrete memory behavior. -/
  | .gld => ∀ {w : Nat} (read : BitVec 64 → BitVec w) (a : BitVec 64),
      CMem.gld read a = Ptx.gld read a
  /- PTX definition: `st.global` updates the addressed value; GX's fake-address guard is runtime policy. -/
  /- Lean modeling: quantifying the prior memory function isolates equality of GX's guard/update policy from concrete memory effects. -/
  | .gst => ∀ {w : Nat} (mem : BitVec 64 → BitVec w) (a : BitVec 64) (v : BitVec w),
      CMem.gst mem a v = Ptx.gst mem a v
  /- PTX definition: `shf.r.b32` right-shifts `b:a` and returns its low word. -/
  /- Lean modeling: the `n ≤ 32` premise covers every amount produced by the modeled wrap and clamp selectors. -/
  | .shfR => ∀ (lo hi : BitVec 32) (n : Nat), n ≤ 32 →
      CBits.shfR lo hi n = Ptx.shfR lo hi n
  /- PTX definition: `shf.l.b32` left-shifts `b:a` and returns its high word. -/
  /- Lean modeling: the `n ≤ 32` premise covers every amount produced by the modeled wrap and clamp selectors. -/
  | .shfL => ∀ (lo hi : BitVec 32) (n : Nat), n ≤ 32 →
      CBits.shfL lo hi n = Ptx.shfL lo hi n
  /- PTX definition: `.wrap` derives the shift amount from the low five control bits. -/
  /- Lean modeling: the equality directly compares C-side masking by 31 with PTX reduction modulo 32. -/
  | .shfAmtWrap => ∀ (c : BitVec 32), CBits.shfAmtWrap c = Ptx.shfAmt true c
  /- PTX definition: `.clamp` limits the unsigned control amount to 32. -/
  /- Lean modeling: the equality compares the C-side conditional with the same natural minimum used by the PTX model. -/
  | .shfAmtClamp => ∀ (c : BitVec 32), CBits.shfAmtClamp c = Ptx.shfAmt false c
  /- PTX definition: `bfi` takes bits from `ins` inside the selected field and from `base` outside it. -/
  /- Lean modeling: the in-range bit index premise states extensional correctness while keeping helper normalization explicit. -/
  | .bfi => ∀ (w : Nat) (ins base : BitVec w) (pos len i : Nat), i < w →
      (CBits.bfi w ins base pos len).getLsbD i =
        if pos ≤ i ∧ i < pos + len then ins.getLsbD (i - pos) else base.getLsbD i
  /- PTX definition: `bfe.uN` copies in-field bits from `pos` and zero-fills the remainder. -/
  /- Lean modeling: this per-bit proposition includes the required low-eight-bit normalization of both raw `.u32` controls. -/
  | .bfeU => ∀ (w : Nat) (a : BitVec w) (pos len i : Nat), i < w →
      (CBits.bfeU w a pos len).getLsbD i =
        if i < len % 256 ∧ pos % 256 + i < w
        then a.getLsbD (pos % 256 + i) else false
  /- PTX definition: generic `prmt` uses four selector nibbles to choose bytes from `b:a`. -/
  /- Lean modeling: the proposition compares complete 32-bit results for generic mode; specialized permutation modes remain outside it. -/
  | .prmt => ∀ (a b sel : BitVec 32), CBits.prmt a b sel = Ptx.prmt a b sel

/- PTX definition: every listed emission kind must implement the corresponding PTX value rule. -/
/- Lean modeling: each exhaustive arm supplies the already-checked equivalence theorem, so an unproved new `EmitKind` cannot elaborate. -/
theorem EmitKind.certified : ∀ k : EmitKind, k.spec
  /- PTX definition: integer `cvt` extends according to source signedness or truncates. -/
  /- Lean modeling: `cvtI_equiv` is quantified over the same signedness, widths, and source value required by `.cvtII.spec`. -/
  | .cvtII => CCvt.cvtI_equiv
  /- PTX definition: scalar `mov.bN` copies its source bit pattern unchanged. -/
  /- Lean modeling: `movB_equiv` directly inhabits the width-generic identity obligation. -/
  | .movB => CMov.movB_equiv
  /- PTX definition: brace-source `mov` places its first element low and second high. -/
  /- Lean modeling: `pack_equiv` directly proves the required equality for arbitrary equal-width halves. -/
  | .movPack => CMov.pack_equiv
  /- PTX definition: the first brace destination receives the packed source's low half. -/
  /- Lean modeling: `unpackLo_equiv` directly proves the dependent `BitVec (h + h)` extraction obligation. -/
  | .movUnpackLo => CMov.unpackLo_equiv
  /- PTX definition: the second brace destination receives the packed source's high half. -/
  /- Lean modeling: `unpackHi_equiv` directly proves the dependent `BitVec (h + h)` extraction obligation. -/
  | .movUnpackHi => CMov.unpackHi_equiv
  /- PTX definition: `mul.wide.uN` is legal only for N = 16 or 32. -/
  /- Lean modeling: the theorem inhabits a generic algebraic proposition; PTX certification is limited to those legal specializations. -/
  | .mulWideU => CW.mulWideU_equiv
  /- PTX definition: `mul.wide.sN` is legal only for N = 16 or 32. -/
  /- Lean modeling: the theorem inhabits a generic algebraic proposition; PTX certification is limited to those legal specializations. -/
  | .mulWideS => CW.mulWideS_equiv
  /- PTX definition: `mul.hi.uN` is legal only for N = 16, 32, or 64. -/
  /- Lean modeling: the theorem proves a generic algebraic equality; PTX certification is limited to those legal specializations. -/
  | .mulHiU => CW.mulHiU_equiv
  /- PTX definition: `mul.hi.sN` is legal only for N = 16, 32, or 64. -/
  /- Lean modeling: the theorem proves a generic algebraic equality; PTX certification is limited to those legal specializations. -/
  | .mulHiS => CW.mulHiS_equiv
  /- PTX definition: `mad.hi.uN` is legal only for N = 16, 32, or 64. -/
  /- Lean modeling: the theorem proves a generic algebraic equality; PTX certification is limited to those legal specializations. -/
  | .madHiU => CW.madHiU_equiv
  /- PTX definition: `mad.hi.sN` is legal only for N = 16, 32, or 64. -/
  /- Lean modeling: the theorem proves a generic algebraic equality; PTX certification is limited to those legal specializations. -/
  | .madHiS => CW.madHiS_equiv
  /- PTX definition: `mad.wide.uN` is legal only for N = 16 or 32. -/
  /- Lean modeling: the theorem proves the generic algebraic equality; explicit wrappers provide the legal PTX specializations. -/
  | .madWideU => CW.madWideU_equiv
  /- PTX definition: `mad.wide.sN` is legal only for N = 16 or 32. -/
  /- Lean modeling: the theorem proves the generic algebraic equality; explicit wrappers provide the legal PTX specializations. -/
  | .madWideS => CW.madWideS_equiv
  /- PTX definition: vector element addresses add a scaled byte offset to the base. -/
  /- Lean modeling: `elemAddr_equiv` directly proves equality for the same base, element index, and byte size. -/
  | .ldStElemAddr => CMem.elemAddr_equiv
  /- PTX definition: `ld.global` reads the addressed value; fake-address handling is GX policy. -/
  /- Lean modeling: `gld_equiv` proves the exact abstract-read result equality without asserting concrete memory semantics. -/
  | .gld => CMem.gld_equiv
  /- PTX definition: `st.global` updates the addressed value; fake-address handling is GX policy. -/
  /- Lean modeling: `gst_equiv` proves the exact abstract-memory-function equality without asserting concrete memory effects. -/
  | .gst => CMem.gst_equiv
  /- PTX definition: `shf.r.b32` right-shifts the concatenated source and returns its low word. -/
  /- Lean modeling: `shfR_equiv` accepts precisely the `n ≤ 32` premise required by this constructor's proposition. -/
  | .shfR => CBits.shfR_equiv
  /- PTX definition: `shf.l.b32` left-shifts the concatenated source and returns its high word. -/
  /- Lean modeling: `shfL_equiv` accepts precisely the `n ≤ 32` premise required by this constructor's proposition. -/
  | .shfL => CBits.shfL_equiv
  /- PTX definition: `.wrap` selects the amount from the low five control bits. -/
  /- Lean modeling: `shfAmtWrap_equiv` directly proves masking and modulo-32 select the same natural amount. -/
  | .shfAmtWrap => CBits.shfAmtWrap_equiv
  /- PTX definition: `.clamp` limits the unsigned control value to 32. -/
  /- Lean modeling: `shfAmtClamp_equiv` directly proves the conditional and PTX minimum produce the same amount. -/
  | .shfAmtClamp => CBits.shfAmtClamp_equiv
  /- PTX definition: `bfi` inserts source-field bits and preserves base bits outside the field. -/
  /- Lean modeling: `bfi_spec` has the identical per-bit parameters, bound, and conditional required by this arm. -/
  | .bfi => CBits.bfi_spec
  /- PTX definition: `bfe.uN` copies selected bits and zero-fills remaining output bits. -/
  /- Lean modeling: `bfeU_spec` has the identical normalized-control per-bit conditional required here. -/
  | .bfeU => CBits.bfeU_spec
  /- PTX definition: generic `prmt.b32` assembles four selector-chosen bytes from `b:a`. -/
  /- Lean modeling: `prmt_equiv` directly proves whole-result equality for the three 32-bit operands of this arm. -/
  | .prmt => CBits.prmt_equiv

/- PTX definition: the certificate index covers the 23 non-template shapes currently modeled. -/
/- Lean modeling: the explicit list supports deterministic runtime enumeration while `spec` and `certified` remain the compile-time totality guarantees. -/
def EmitKind.all : List EmitKind :=
  [.cvtII, .movB, .movPack, .movUnpackLo, .movUnpackHi,
   .mulWideU, .mulWideS, .mulHiU, .mulHiS,
   .madHiU, .madHiS, .madWideU, .madWideS,
   .ldStElemAddr, .gld, .gst,
   .shfR, .shfL, .shfAmtWrap, .shfAmtClamp,
   .bfi, .bfeU, .prmt]

/- PTX definition: each modeled emission kind has a corresponding Lean equivalence theorem. -/
/- Lean modeling: this total name mapping lets `gxcheck` resolve and independently audit the exact theorem cited by every reporting constructor. -/
def EmitKind.thmName : EmitKind → String
  | .cvtII => "Gxptx.CCvt.cvtI_equiv"
  | .movB => "Gxptx.CMov.movB_equiv"
  | .movPack => "Gxptx.CMov.pack_equiv"
  | .movUnpackLo => "Gxptx.CMov.unpackLo_equiv"
  | .movUnpackHi => "Gxptx.CMov.unpackHi_equiv"
  | .mulWideU => "Gxptx.CW.mulWideU_equiv"
  | .mulWideS => "Gxptx.CW.mulWideS_equiv"
  | .mulHiU => "Gxptx.CW.mulHiU_equiv"
  | .mulHiS => "Gxptx.CW.mulHiS_equiv"
  | .madHiU => "Gxptx.CW.madHiU_equiv"
  | .madHiS => "Gxptx.CW.madHiS_equiv"
  | .madWideU => "Gxptx.CW.madWideU_equiv"
  | .madWideS => "Gxptx.CW.madWideS_equiv"
  | .ldStElemAddr => "Gxptx.CMem.elemAddr_equiv"
  | .gld => "Gxptx.CMem.gld_equiv"
  | .gst => "Gxptx.CMem.gst_equiv"
  | .shfR => "Gxptx.CBits.shfR_equiv"
  | .shfL => "Gxptx.CBits.shfL_equiv"
  | .shfAmtWrap => "Gxptx.CBits.shfAmtWrap_equiv"
  | .shfAmtClamp => "Gxptx.CBits.shfAmtClamp_equiv"
  | .bfi => "Gxptx.CBits.bfi_spec"
  | .bfeU => "Gxptx.CBits.bfeU_spec"
  | .prmt => "Gxptx.CBits.prmt_equiv"

end Gxptx
