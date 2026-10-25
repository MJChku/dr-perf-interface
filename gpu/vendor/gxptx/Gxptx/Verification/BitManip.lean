/-
  Proofs connecting GX's shallow C bit-helper models to the corresponding PTX
  value semantics. The helper lemmas preserve the choice-free proof strategy
  and axiom budget of the former `CSem.lean` section.
-/
import Gxptx.Semantics.PTX.BitManip
import Gxptx.Semantics.C.BitManip
import Gxptx.Verification.Move

namespace Gxptx

/- PTX definition: bit-field masks contain ones exactly below their selected length. -/
/- Lean modeling: this choice-free bit lemma characterizes `2^n - 1` without importing the library theorem's stronger classical axiom dependency. -/
private theorem testBit_ones (n i : Nat) :
    Nat.testBit (2 ^ n - 1) i = decide (i < n) := by
  rw [Nat.testBit_to_div_mod]
  by_cases h : i < n
  · have hpow : 2 ^ n = 2 ^ i * 2 ^ (n - i) := by
      rw [← Nat.pow_add]
      congr 1
      omega
    have h2i : 0 < 2 ^ i := Nat.two_pow_pos i
    have hsplit : 2 ^ (n - i) = 2 * 2 ^ (n - i - 1) := by
      have hni : n - i = (n - i - 1) + 1 := by omega
      rw [hni, Nat.pow_succ, Nat.mul_comm]
      have hred : n - i - 1 + 1 - 1 = n - i - 1 := by omega
      rw [hred]
    have hmul : 2 ^ i * (2 ^ (n - i) - 1) + 2 ^ i = 2 ^ i * 2 ^ (n - i) := by
      rw [← Nat.mul_succ]
      congr 1
      have := Nat.two_pow_pos (n - i)
      omega
    have heq : 2 ^ n - 1 = 2 ^ i * (2 ^ (n - i) - 1) + (2 ^ i - 1) := by
      omega
    have hdiv : (2 ^ n - 1) / 2 ^ i = 2 ^ (n - i) - 1 := by
      rw [heq, Nat.mul_add_div h2i, Nat.div_eq_of_lt (by omega)]
      omega
    have hodd : (2 ^ (n - i) - 1) % 2 = 1 := by
      have := Nat.two_pow_pos (n - i - 1)
      omega
    rw [hdiv]
    simp [hodd, h]
  · have hge : 2 ^ n ≤ 2 ^ i := Nat.pow_le_pow_right (by decide) (by omega)
    have h2n := Nat.two_pow_pos n
    have h0 : (2 ^ n - 1) / 2 ^ i = 0 := Nat.div_eq_of_lt (by omega)
    rw [h0]
    simp [h]

/- PTX definition: an N-bit all-ones mask has every in-range bit set and every out-of-range read clear. -/
/- Lean modeling: reducing `BitVec.allOnes` to `testBit_ones` preserves the original choice-free proof used by the field-operation verification. -/
private theorem getLsbD_allOnes' {v : Nat} (i : Nat) :
    (BitVec.allOnes v).getLsbD i = decide (i < v) := by
  show Nat.testBit (BitVec.allOnes v).toNat i = decide (i < v)
  rw [BitVec.toNat_allOnes]
  exact testBit_ones v i

/- PTX definition: complementing a field mask flips every bit within the instruction width. -/
/- Lean modeling: expanding complement as XOR with all ones and using the previous in-range lemma yields the required choice-free bit identity. -/
private theorem getLsbD_not' {v : Nat} (x : BitVec v) (i : Nat) (h : i < v) :
    (~~~x).getLsbD i = !x.getLsbD i := by
  rw [BitVec.not_def, BitVec.getLsbD_xor, getLsbD_allOnes']
  simp [h]

/- PTX definition: low control bits determine wrapped shift amounts and selector nibbles. -/
/- Lean modeling: bit extensionality proves masking by `2^n-1` equals natural-value reduction modulo `2^n` without classical choice. -/
private theorem and_mask_toNat {w : Nat} (x : BitVec w) (n : Nat) (hn : n < w) :
    (x &&& BitVec.ofNat w (2 ^ n - 1)).toNat = x.toNat % 2 ^ n := by
  have hpw : 2 ^ n < 2 ^ w := Nat.pow_lt_pow_of_lt (by decide : 1 < 2) hn
  have hpn : 0 < 2 ^ n := Nat.two_pow_pos n
  have hlt : 2 ^ n - 1 < 2 ^ w := by omega
  have hmod : x.toNat % 2 ^ n < 2 ^ w := by
    have h1 : x.toNat % 2 ^ n < 2 ^ n := Nat.mod_lt _ hpn
    omega
  have hbv : x &&& BitVec.ofNat w (2 ^ n - 1)
      = BitVec.ofNat w (x.toNat % 2 ^ n) := by
    apply BitVec.eq_of_getLsbD_eq
    intro ⟨j, hj⟩
    show (x &&& BitVec.ofNat w (2 ^ n - 1)).getLsbD j
        = (BitVec.ofNat w (x.toNat % 2 ^ n)).getLsbD j
    have hL : (BitVec.ofNat w (2 ^ n - 1)).getLsbD j = decide (j < n) := by
      show Nat.testBit ((2 ^ n - 1) % 2 ^ w) j = decide (j < n)
      rw [Nat.mod_eq_of_lt hlt]
      exact testBit_ones n j
    have hR : (BitVec.ofNat w (x.toNat % 2 ^ n)).getLsbD j
        = (decide (j < n) && x.getLsbD j) := by
      show Nat.testBit (x.toNat % 2 ^ n % 2 ^ w) j = _
      rw [Nat.mod_eq_of_lt hmod, Nat.testBit_mod_two_pow]
      rfl
    rw [BitVec.getLsbD_and, hL, hR, Bool.and_comm]
  rw [hbv]
  show x.toNat % 2 ^ n % 2 ^ w = x.toNat % 2 ^ n
  exact Nat.mod_eq_of_lt hmod

namespace CBits

/- PTX definition: `.wrap` computes the funnel-shift amount from the low five bits of `c`. -/
/- Lean modeling: `and_mask_toNat` proves the C-side value model's `c & 31` result equals the shallow PTX modulo-32 amount. -/
theorem shfAmtWrap_equiv (c : BitVec 32) : shfAmtWrap c = Ptx.shfAmt true c := by
  show (c &&& 31#32).toNat = c.toNat % 32
  exact and_mask_toNat c 5 (by omega)

/- PTX definition: `.clamp` computes `min(c, 32)` as the funnel-shift amount. -/
/- Lean modeling: splitting on the unsigned comparison with 32 proves the C-side value-model conditional and PTX natural minimum return the same amount. -/
theorem shfAmtClamp_equiv (c : BitVec 32) : shfAmtClamp c = Ptx.shfAmt false c := by
  have h32 : (32#32).toNat = 32 := rfl
  show (if 32#32 < c then 32 else c.toNat) = min c.toNat 32
  by_cases h : 32#32 < c
  · have hlt := BitVec.lt_def.mp h
    rw [if_pos h]
    omega
  · have hle : ¬ (32:Nat) < c.toNat := fun hc =>
      h (BitVec.lt_def.mpr (by omega))
    rw [if_neg h]
    omega

/- PTX definition: `shf` treats `b:a` as one 64-bit concatenated source with `a` in its low word. -/
/- Lean modeling: the append bit rule separates low-word and high-word reads for the extensional funnel-shift proofs. -/
private theorem getLsbD_append32 (hi lo : BitVec 32) (j : Nat) :
    (hi ++ lo).getLsbD j
      = if j < 32 then lo.getLsbD j else hi.getLsbD (j - 32) := by
  rw [BitVec.getLsbD_append]
  by_cases h : j < 32 <;> simp [h]

/- PTX definition: `shf.r.b32` returns the low word of `b:a` shifted right by the selected amount. -/
/- Lean modeling: per-bit analysis proves the C-side `shfR` value model equals PTX concatenation for every amount produced by wrap or clamp. -/
theorem shfR_equiv (lo hi : BitVec 32) (n : Nat) (hn : n ≤ 32) :
    shfR lo hi n = Ptx.shfR lo hi n := by
  apply BitVec.eq_of_getLsbD_eq
  intro ⟨i, hilt⟩
  have hrhs : (Ptx.shfR lo hi n).getLsbD i
      = if n + i < 32 then lo.getLsbD (n + i) else hi.getLsbD (n + i - 32) := by
    simp only [Ptx.shfR, BitVec.getLsbD_extractLsb', BitVec.getLsbD_ushiftRight,
      Nat.zero_add, getLsbD_append32, hilt, decide_true, Bool.true_and]
  rw [hrhs]
  unfold shfR
  by_cases h0 : n = 0
  · subst h0
    rw [if_pos rfl, if_pos (by omega : 0 + i < 32), Nat.zero_add]
  · rw [if_neg h0]
    by_cases h32 : 32 ≤ n
    · have hn32 : n = 32 := by omega
      subst hn32
      rw [if_pos (by omega : (32:Nat) ≤ 32), if_neg (by omega : ¬ 32 + i < 32)]
      simp only [Nat.sub_self, BitVec.getLsbD_ushiftRight, Nat.zero_add]
      congr 1
      omega
    · rw [if_neg h32]
      simp only [BitVec.getLsbD_or, BitVec.getLsbD_ushiftRight,
        BitVec.getLsbD_shiftLeft, hilt, decide_true, Bool.true_and]
      by_cases hc : n + i < 32
      · have hlt : i < 32 - n := by omega
        rw [if_pos hc]
        simp [hlt]
      · have h1 : lo.getLsbD (n + i) = false := BitVec.getLsbD_ge lo (n + i) (by omega)
        have h2 : ¬ i < 32 - n := by omega
        have h3 : i - (32 - n) = n + i - 32 := by omega
        rw [if_neg hc]
        simp [h1, h2, h3]

/- PTX definition: `shf.l.b32` returns the high word of `b:a` shifted left by the selected amount. -/
/- Lean modeling: per-bit analysis proves the C-side `shfL` value model equals PTX concatenation for every amount produced by wrap or clamp. -/
theorem shfL_equiv (lo hi : BitVec 32) (n : Nat) (hn : n ≤ 32) :
    shfL lo hi n = Ptx.shfL lo hi n := by
  apply BitVec.eq_of_getLsbD_eq
  intro ⟨i, hilt⟩
  show (shfL lo hi n).getLsbD i = (Ptx.shfL lo hi n).getLsbD i
  have h64 : 32 + i < 64 := by omega
  have hge : ¬ (32 + i < n) := by omega
  have hrhs : (Ptx.shfL lo hi n).getLsbD i
      = if 32 + i - n < 32 then lo.getLsbD (32 + i - n)
        else hi.getLsbD (32 + i - n - 32) := by
    simp only [Ptx.shfL, BitVec.getLsbD_extractLsb', BitVec.getLsbD_shiftLeft,
      getLsbD_append32, hilt, decide_true, Bool.true_and]
    simp [h64, hge]
  rw [hrhs]
  unfold shfL
  by_cases h0 : n = 0
  · subst h0
    rw [if_pos rfl, if_neg (by omega : ¬ 32 + i - 0 < 32)]
    congr 1
    omega
  · rw [if_neg h0]
    by_cases h32 : 32 ≤ n
    · have hn32 : n = 32 := by omega
      subst hn32
      rw [if_pos (by omega : (32:Nat) ≤ 32), if_pos (by omega : 32 + i - 32 < 32)]
      simp only [Nat.sub_self, BitVec.getLsbD_shiftLeft, hilt, decide_true,
        Bool.true_and, Nat.sub_zero]
      have h1 : ¬ i < 0 := by omega
      simp [h1]
      congr 1
      omega
    · rw [if_neg h32]
      simp only [BitVec.getLsbD_or, BitVec.getLsbD_shiftLeft,
        BitVec.getLsbD_ushiftRight, hilt, decide_true, Bool.true_and]
      by_cases hc : 32 + i - n < 32
      · have hin : i < n := by omega
        have heq : 32 - n + i = 32 + i - n := by omega
        rw [if_pos hc]
        simp [hin, heq]
      · have hin : ¬ i < n := by omega
        have h1 : lo.getLsbD (32 - n + i) = false :=
          BitVec.getLsbD_ge lo _ (by omega)
        have heq : i - n = 32 + i - n - 32 := by omega
        rw [if_neg hc]
        simp [hin, h1]
        rw [heq]

/- PTX definition: the modeled low-field mask has bit `i` set exactly when `i < len`. -/
/- Lean modeling: the in-range complement and shift lemmas reduce the C-side mask value expression to the per-bit rule used by both field proofs. -/
theorem mask_getLsbD {w : Nat} (len i : Nat) (h : i < w) :
    (mask w len).getLsbD i = decide (i < len) := by
  rw [mask, getLsbD_not' _ _ h]
  simp only [BitVec.getLsbD_shiftLeft, getLsbD_allOnes', h, decide_true,
    Bool.true_and]
  by_cases hl : i < len
  · simp [hl]
  · simp [hl]
    omega

/- PTX definition: `bfi.bN` writes source bit `i-pos` exactly inside the selected field and otherwise preserves base bit `i`. -/
/- Lean modeling: exhaustive position and length cases prove the current GX helper model implements the ISA's per-bit insertion rule after helper-side 8-bit normalization. -/
theorem bfi_spec (w : Nat) (ins base : BitVec w) (pos len i : Nat)
    (hi : i < w) :
    (bfi w ins base pos len).getLsbD i =
      if pos ≤ i ∧ i < pos + len then ins.getLsbD (i - pos)
      else base.getLsbD i := by
  by_cases hp : w ≤ pos
  · rw [bfi, if_pos hp, if_neg (by omega)]
  · rw [bfi, if_neg hp]
    have hnot : ∀ y : BitVec w, (~~~y).getLsbD i = !y.getLsbD i :=
      fun y => getLsbD_not' y i hi
    simp only [BitVec.getLsbD_or, BitVec.getLsbD_and, hnot,
      BitVec.getLsbD_shiftLeft, hi, decide_true, Bool.true_and]
    -- NOTE: split on the ATOMS, not the conjunction — `omega` on a
    -- `¬(A ∧ B)` hypothesis pulls in Classical.choice.
    by_cases hpos : pos ≤ i
    · by_cases hlen : i < pos + len
      · -- inside the field
        have h2 : i - pos < len := by omega
        have hm : (if w ≤ len then BitVec.allOnes w else mask w len).getLsbD (i - pos) = true := by
          by_cases hl : w ≤ len
          · rw [if_pos hl, getLsbD_allOnes']
            exact decide_eq_true (by omega)
          · rw [if_neg hl, mask_getLsbD _ _ (by omega)]
            exact decide_eq_true h2
        have hd : decide (i < pos) = false := decide_eq_false (by omega)
        have hins : pos ≤ i ∧ i < pos + len := ⟨hpos, hlen⟩
        rw [if_pos hins, hm, hd]
        simp only [Bool.not_false, Bool.and_true, Bool.true_and, Bool.not_true,
          Bool.and_false, Bool.false_and, Bool.or_false, Bool.false_or]
      · -- right of the field: i - pos ≥ len
        have h2 : ¬ (i - pos < len) := by omega
        have hlw : ¬ (w ≤ len) := by omega
        have hm : (if w ≤ len then BitVec.allOnes w else mask w len).getLsbD (i - pos) = false := by
          rw [if_neg hlw, mask_getLsbD _ _ (by omega)]
          exact decide_eq_false h2
        have hins : ¬ (pos ≤ i ∧ i < pos + len) := fun hc => hlen hc.2
        rw [if_neg hins, hm]
        simp only [Bool.and_false, Bool.not_false, Bool.and_true, Bool.or_false]
    · -- left of the field: i < pos
      have hd : decide (i < pos) = true := decide_eq_true (by omega)
      have hins : ¬ (pos ≤ i ∧ i < pos + len) := fun hc => hpos hc.1
      rw [if_neg hins, hd]
      simp only [Bool.not_true, Bool.false_and, Bool.and_false, Bool.not_false,
        Bool.and_true, Bool.or_false]

/- PTX definition: for already-normalized controls, unsigned `bfe.uN` copies source bit `pos+i` only while it lies inside the requested field and zero-fills all other bits. -/
/- Lean modeling: this internal lemma proves the branch-exact unsigned core's per-bit result before the public wrapper applies low-eight-bit normalization. -/
private theorem bfeUCore_spec (w : Nat) (a : BitVec w) (pos len i : Nat) (hi : i < w) :
    (bfeUCore w a pos len).getLsbD i =
      if i < len ∧ pos + i < w then a.getLsbD (pos + i) else false := by
  by_cases h0 : len = 0
  · subst h0
    rw [bfeUCore, if_pos rfl, if_neg (by omega)]
    exact BitVec.getLsbD_zero
  · rw [bfeUCore, if_neg h0]
    by_cases hp : w ≤ pos
    · rw [if_pos hp, if_neg (by omega)]
      exact BitVec.getLsbD_zero
    · rw [if_neg hp]
      -- split on the ATOMS (omega on ¬(A ∧ B) is classical)
      by_cases hl : w ≤ len
      · rw [if_pos hl]
        simp only [BitVec.getLsbD_ushiftRight]
        by_cases hil : i < len
        · by_cases hpw : pos + i < w
          · have hc : i < len ∧ pos + i < w := ⟨hil, hpw⟩
            rw [if_pos hc]
          · have hc : ¬ (i < len ∧ pos + i < w) := fun hc => hpw hc.2
            rw [if_neg hc]
            exact BitVec.getLsbD_ge a (pos + i) (by omega)
        · exact absurd (by omega : i < len) hil
      · rw [if_neg hl]
        simp only [BitVec.getLsbD_and, BitVec.getLsbD_ushiftRight,
          mask_getLsbD _ _ hi]
        by_cases hil : i < len
        · by_cases hpw : pos + i < w
          · have hc : i < len ∧ pos + i < w := ⟨hil, hpw⟩
            rw [if_pos hc, decide_eq_true hil, Bool.and_true]
          · have hc : ¬ (i < len ∧ pos + i < w) := fun hc => hpw hc.2
            rw [if_neg hc]
            rw [BitVec.getLsbD_ge a (pos + i) (by omega), Bool.false_and]
        · have hc : ¬ (i < len ∧ pos + i < w) := fun hc => hil hc.1
          rw [if_neg hc, decide_eq_false hil, Bool.and_false]

/- PTX definition: `bfe` normalizes both `.u32` controls with `& 0xff` before unsigned extraction. -/
/- Lean modeling: unfolding the wrapper and applying `bfeUCore_spec` proves the complete per-bit unsigned rule for raw control operands. -/
theorem bfeU_spec (w : Nat) (a : BitVec w) (pos len i : Nat) (hi : i < w) :
    (bfeU w a pos len).getLsbD i =
      if i < len % 256 ∧ pos % 256 + i < w
      then a.getLsbD (pos % 256 + i) else false := by
  unfold bfeU
  exact bfeUCore_spec w a (pos % 256) (len % 256) i hi

/- PTX definition: signed `bfe.sN` copies in-range field bits and otherwise repeats the selected field sign bit; length zero is all zero. -/
/- Lean modeling: the proof combines the unsigned-core bit theorem with the copied-range mask, including fields that cross or start beyond the source MSB. -/
theorem bfeS_spec (w : Nat) (a : BitVec w) (pos len i : Nat) (hi : i < w) :
    let p := pos % 256
    let l := len % 256
    (bfeS w a pos len).getLsbD i =
      if l = 0 then false
      else if i < l ∧ p + i < w then a.getLsbD (p + i)
      else a.getLsbD (min (p + l - 1) (w - 1)) := by
  dsimp only
  unfold bfeS
  dsimp only
  let p := pos % 256
  let l := len % 256
  let copied := min l (w - p)
  let sbit := a.getLsbD (min (p + l - 1) (w - 1))
  by_cases h0 : l = 0
  · rw [if_pos h0, if_pos h0]
    exact BitVec.getLsbD_zero
  · rw [if_neg h0, if_neg h0]
    have hcore := bfeUCore_spec w a p copied i hi
    have hcond :
        (i < copied ∧ p + i < w) ↔ (i < l ∧ p + i < w) := by
      constructor
      · intro hc
        have hmin : i < min l (w - p) := by
          simpa [copied] using hc.1
        exact ⟨(Nat.lt_min.mp hmin).1, hc.2⟩
      · intro hc
        have hsub : i < w - p := by omega
        have hmin : i < min l (w - p) := Nat.lt_min.mpr ⟨hc.1, hsub⟩
        exact ⟨by simpa [copied] using hmin, hc.2⟩
    by_cases hs : sbit
    · rw [if_pos hs]
      simp only [BitVec.getLsbD_or, getLsbD_not' _ i hi,
        mask_getLsbD _ i hi, hcore]
      by_cases hf : i < l ∧ p + i < w
      · have hc : i < copied ∧ p + i < w := hcond.mpr hf
        have hic : i < copied := hc.1
        rw [if_pos hf, if_pos hc, decide_eq_true hic]
        simp
      · have hc : ¬ (i < copied ∧ p + i < w) := fun hc => hf (hcond.mp hc)
        have hic : ¬ i < copied := by
          intro hic
          apply hf
          apply hcond.mp
          exact ⟨hic, by omega⟩
        rw [if_neg hf, if_neg hc, decide_eq_false hic]
        change true = sbit
        exact hs.symm
    · rw [if_neg hs, hcore]
      by_cases hf : i < l ∧ p + i < w
      · have hc : i < copied ∧ p + i < w := hcond.mpr hf
        rw [if_pos hf, if_pos hc]
      · have hc : ¬ (i < copied ∧ p + i < w) := fun hc => hf (hcond.mp hc)
        rw [if_neg hf, if_neg hc]
        change false = sbit
        cases hsb : sbit
        · rfl
        · exact (hs hsb).elim

end CBits

/- PTX definition: generic `prmt` uses selector bit 3 to test and replicate a selected byte's sign bit at bit 7. -/
/- Lean modeling: this choice-free literal-bit lemma isolates bit 7 of `0x80`, supporting the C-side value model's sign-bit test proof. -/
private theorem getLsbD_0x80 (k : Nat) : (128#32).getLsbD k = decide (7 = k) := by
  show Nat.testBit (128 % 2 ^ 32) k = decide (7 = k)
  rw [Nat.mod_eq_of_lt (by omega), show (128 : Nat) = 2 ^ 7 from rfl,
    Nat.testBit_two_pow]

/- PTX definition: masking a selected byte with `0x80` retains only its sign bit. -/
/- Lean modeling: bit extensionality and `getLsbD_0x80` prove the C-side modeled mask is either exactly `0x80` or zero. -/
private theorem and_0x80 (x : BitVec 32) :
    x &&& 128#32 = if x.getLsbD 7 then 128#32 else 0#32 := by
  apply BitVec.eq_of_getLsbD_eq
  intro ⟨j, hj⟩
  show (x &&& 128#32).getLsbD j
      = (if x.getLsbD 7 then 128#32 else 0#32).getLsbD j
  rw [BitVec.getLsbD_and, getLsbD_0x80]
  by_cases h7 : x.getLsbD 7
  · rw [if_pos h7]
    rw [getLsbD_0x80]
    by_cases hj7 : (7 : Nat) = j
    · subst hj7
      simp [h7]
    · simp [hj7]
  · rw [if_neg h7]
    simp only [BitVec.getLsbD_zero, Bool.and_eq_false_imp, decide_eq_true_eq]
    intro hb
    by_cases hj7 : (7 : Nat) = j
    · subst hj7
      exact absurd hb h7
    · simp [hj7]

/- PTX definition: generic `prmt` branches on whether the selected byte's sign bit is set. -/
/- Lean modeling: rewriting the isolated `0x80` result proves the C-side value model's nonzero comparison is equivalent to reading source bit 7. -/
private theorem and_0x80_ne_zero (x : BitVec 32) :
    ((x &&& 128#32) ≠ 0#32) ↔ (x.getLsbD 7 = true) := by
  rw [and_0x80]
  by_cases h : x.getLsbD 7
  · rw [if_pos h]
    exact iff_of_true (by decide) h
  · rw [if_neg h]
    exact iff_of_false (fun hc => hc rfl) h

namespace CBits

/- PTX definition: each generic `prmt` selector nibble chooses one byte from `b:a` and optionally sign-replicates it. -/
/- Lean modeling: the bit-mask and sign-test lemmas prove one C-side `prmtByte` value step equals the PTX-selected byte extended to 32 bits. -/
private theorem prmtByte_equiv (src : BitVec 64) (s : Nat) :
    prmtByte src s = (Ptx.prmtSel src s).setWidth 32 := by
  have hbyte : (src >>> ((s % 8) * 8)).setWidth 32 &&& 255#32
      = (src.extractLsb' ((s % 8) * 8) 8).setWidth 32 := by
    apply BitVec.eq_of_getLsbD_eq
    intro ⟨j, hj⟩
    show ((src >>> ((s % 8) * 8)).setWidth 32 &&& 255#32).getLsbD j
        = ((src.extractLsb' ((s % 8) * 8) 8).setWidth 32).getLsbD j
    have h255 : (255#32).getLsbD j = decide (j < 8) := by
      show Nat.testBit (255 % 2 ^ 32) j = decide (j < 8)
      rw [Nat.mod_eq_of_lt (by omega), show (255 : Nat) = 2 ^ 8 - 1 from rfl]
      exact testBit_ones 8 j
    simp only [BitVec.getLsbD_and, BitVec.getLsbD_setWidth,
      BitVec.getLsbD_ushiftRight, BitVec.getLsbD_extractLsb', h255, hj,
      decide_true, Bool.true_and]
    by_cases h8 : j < 8 <;> simp [h8, Bool.and_comm]
  unfold prmtByte Ptx.prmtSel
  rw [hbyte]
  by_cases hs : 8 ≤ s % 16
  · rw [if_pos hs, if_pos hs]
    have hcond : (((src.extractLsb' ((s % 8) * 8) 8).setWidth 32 &&& 128#32)
          ≠ 0#32)
        ↔ ((src.extractLsb' ((s % 8) * 8) 8).getLsbD 7 = true) := by
      rw [and_0x80_ne_zero]
      simp [BitVec.getLsbD_setWidth]
    by_cases h7 : (src.extractLsb' ((s % 8) * 8) 8).getLsbD 7 = true
    · rw [if_pos (hcond.mpr h7), if_pos h7]
      decide
    · rw [if_neg (fun hc => h7 (hcond.mp hc)), if_neg h7]
      decide
  · rw [if_neg hs, if_neg hs]

/- PTX definition: generic `prmt` places selector outputs into destination bytes zero through three in little-endian order. -/
/- Lean modeling: bit extensionality proves the C-side value model's shift-and-or assembly equals PTX high-to-low concatenation of the same four bytes. -/
private theorem bytes_assemble (b0 b1 b2 b3 : BitVec 8) :
    b0.setWidth 32 ||| (b1.setWidth 32 <<< 8) ||| (b2.setWidth 32 <<< 16)
      ||| (b3.setWidth 32 <<< 24)
      = b3 ++ b2 ++ b1 ++ b0 := by
  apply BitVec.eq_of_getLsbD_eq
  intro ⟨i, hilt⟩
  show (b0.setWidth 32 ||| (b1.setWidth 32 <<< 8) ||| (b2.setWidth 32 <<< 16)
      ||| (b3.setWidth 32 <<< 24)).getLsbD i
      = (b3 ++ b2 ++ b1 ++ b0).getLsbD i
  rw [BitVec.getLsbD_append, BitVec.getLsbD_append, BitVec.getLsbD_append]
  simp only [BitVec.getLsbD_or, BitVec.getLsbD_setWidth,
    BitVec.getLsbD_shiftLeft, hilt, decide_true, Bool.true_and]
  by_cases c0 : i < 8
  · have f1 : i < 16 := by omega
    have f2 : i < 24 := by omega
    simp [c0, f1, f2]
  · by_cases c1 : i < 16
    · have f0 : b0.getLsbD i = false := BitVec.getLsbD_ge b0 i (by omega)
      have f1 : i - 8 < 8 := by omega
      have f2 : i < 24 := by omega
      have f3 : i - 8 < 32 := by omega
      simp [c0, c1, f0, f1, f2, f3]
    · by_cases c2 : i < 24
      · have f0 : b0.getLsbD i = false := BitVec.getLsbD_ge b0 i (by omega)
        have f1 : b1.getLsbD (i - 8) = false := BitVec.getLsbD_ge b1 _ (by omega)
        have f2 : ¬ (i - 8 < 8) := by omega
        have f3 : i - 16 < 8 := by omega
        have f4 : i - 16 < 32 := by omega
        have f5 : i - 8 - 8 = i - 16 := by omega
        simp [c0, c1, c2, f0, f1, f2, f3, f4, f5]
      · have f0 : b0.getLsbD i = false := BitVec.getLsbD_ge b0 i (by omega)
        have f1 : b1.getLsbD (i - 8) = false := BitVec.getLsbD_ge b1 _ (by omega)
        have f2 : b2.getLsbD (i - 16) = false := BitVec.getLsbD_ge b2 _ (by omega)
        have f3 : ¬ (i - 8 < 8) := by omega
        have f4 : ¬ (i - 8 - 8 < 8) := by omega
        have f5 : i - 8 - 8 - 8 = i - 24 := by omega
        have f6 : i - 24 < 32 := by omega
        simp [c0, c1, c2, f0, f1, f2, f3, f4, f5, f6]

/- PTX definition: each generic `prmt` control nibble is the corresponding shifted selector value reduced modulo 16. -/
/- Lean modeling: specializing `and_mask_toNat` to four bits proves the C-side modeled `x & 0xf` extraction equals the PTX nibble value. -/
private theorem and_15_toNat (x : BitVec 32) :
    (x &&& 15#32).toNat = x.toNat % 16 := by
  exact and_mask_toNat x 4 (by omega)

/- PTX definition: generic `prmt.b32` assembles the four selector-nibble-selected bytes from `b:a`. -/
/- Lean modeling: source packing, per-byte equivalence, nibble extraction, and byte assembly prove equality for generic mode only; specialized modes remain outside this theorem. -/
theorem prmt_equiv (a b sel : BitVec 32) :
    prmt a b sel = Ptx.prmt a b sel := by
  have hsrc : a.setWidth 64 ||| (b.setWidth 64 <<< 32) = b ++ a :=
    CMov.pack_equiv a b
  unfold prmt Ptx.prmt
  rw [hsrc]
  simp only [prmtByte_equiv, and_15_toNat]
  exact bytes_assemble _ _ _ _

end CBits
end Gxptx
