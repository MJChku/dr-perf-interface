/-
  Finite C-template shapes and their shallow value semantics.

  `CTemplate` is an index of the C expression shapes printed by the emitter,
  not a deep embedding of PTX syntax. A constructor's `t : PW` field selects
  the `(u)int32_t` or `(u)int64_t` casts in the printer. The width-generic
  `evalC` definition intentionally preserves the existing model and currently
  does not inspect `t`; this module therefore does not formalize that the
  printed width `t.bits` equals its implicit Lean value width `w`.
-/
import Gxptx.Semantics.Types
import Gxptx.Semantics.PTX.Arithmetic

namespace Gxptx

/- PTX definition: `.b32`, `.u32`, and `.s32` are 32-bit types, while `.b64`, `.u64`, and `.s64` are 64-bit types. -/
/- Lean modeling: `PW` is the finite 32/64 print-width index used to select the emitted C operand casts. -/
inductive PW | w32 | w64
  deriving Repr, DecidableEq

/- PTX definition: the numeric suffix of each modeled fundamental type fixes its width to 32 or 64 bits. -/
/- Lean modeling: `PW.bits` maps each print-width index to the corresponding C cast width without asserting a link to any implicit `BitVec w`. -/
def PW.bits : PW → Nat
  | .w32 => 32
  | .w64 => 64

/- PTX definition: the modeled arithmetic, comparison, selection, logic, and shift instructions each have a fixed operand/result shape. -/
/- Lean modeling: `CTemplate` is a finite C-shape index; each `t : PW` field selects printed 32/64 casts, but no `PW`↔implicit-`w` relation is formalized here. -/
inductive CTemplate
  /- PTX definition: unsaturated integer `add` retains the destination-width sum. -/
  /- Lean modeling: `t` selects the printer's unsigned cast width; this constructor alone does not certify the printed expression. -/
  | add (t : PW)
  /- PTX definition: `add.f16x2` performs two independent RN-even binary16 additions in a 32-bit packed register. -/
  /- Lean modeling: this fixed-width template selects the deterministic integer-only helper and its exact packed value model. -/
  | addF16x2
  /- PTX definition: unsaturated integer `sub` retains the destination-width difference. -/
  /- Lean modeling: `t` selects the printer's unsigned cast width so wraparound is defined, without linking that width to Lean's implicit `w`. -/
  | sub (t : PW)
  /- PTX definition: `mul.lo` retains the destination-width low product bits. -/
  /- Lean modeling: `t` selects unsigned multiplication in the printer; subword destination conversion remains `gap:subword-template-certificate`. -/
  | mul (t : PW)
  /- PTX definition: `mul.bf16x2` performs two independent RN-even bfloat16 multiplications in a packed 32-bit register. -/
  /- Lean modeling: this fixed-width constructor selects the lane-wise helper and exact packed value model; emitted host behavior remains a documented trust boundary. -/
  | mulBF16x2
  /- PTX definition: `mad.lo` adds `c` to the low product bits at destination width. -/
  /- Lean modeling: the unsigned print width makes both operations wrap, but this shape does not itself prove the emitted assignment boundary. -/
  | mad (t : PW)
  /- PTX definition: `neg` produces the destination-width two's-complement negation. -/
  /- Lean modeling: an unsigned cast makes unary negation defined modulo the print width; the constructor is only a printer index. -/
  | neg (t : PW)
  /- PTX definition: bit-type `not` complements every destination bit. -/
  /- Lean modeling: `t` selects an unsigned fixed-width cast before `~`, while proof about the printed token stream lies outside this tag. -/
  | bnot (t : PW)
  /- PTX definition: bit-type `and` combines corresponding operand bits. -/
  /- Lean modeling: the constructor selects the bitwise operator and cast width but carries no independent ISO C semantics. -/
  | band (t : PW)
  /- PTX definition: bit-type `or` combines corresponding operand bits. -/
  /- Lean modeling: the constructor selects the bitwise operator and cast width but carries no independent ISO C semantics. -/
  | bor (t : PW)
  /- PTX definition: bit-type `xor` combines corresponding operand bits exclusively. -/
  /- Lean modeling: the constructor selects the bitwise operator and cast width but carries no independent ISO C semantics. -/
  | bxor (t : PW)
  /- PTX definition: predicate `not` reverses the source predicate. -/
  /- Lean modeling: C `!` maps zero to one and nonzero to zero, agreeing with GX's canonical predicate value model. -/
  | lnot
  /- PTX definition: predicate `and` is true exactly when both operands are true. -/
  /- Lean modeling: C `&&` normalizes the zero/nonzero tests to 0 or 1; this tag does not model evaluation side effects. -/
  | land
  /- PTX definition: predicate `or` is true when either operand is true. -/
  /- Lean modeling: C `||` normalizes the zero/nonzero tests to 0 or 1; this tag does not model evaluation side effects. -/
  | lor
  /- PTX definition: predicate `xor` is true exactly when its operands differ. -/
  /- Lean modeling: comparing normalized truth values gives predicate xor; the constructor records shape rather than an ISO C evaluator. -/
  | lxor
  /- PTX definition: `shl` zero-fills and clamps an excessive amount to the operand width. -/
  /- Lean modeling: the selected helper guards excessive shifts; `t` remains a print-width index not tied here to implicit `w`. -/
  | shl (t : PW)
  /- PTX definition: unsigned and bit-type `shr` zero-fill and clamp excessive amounts. -/
  /- Lean modeling: the selected helper guards the shift and uses unsigned data, without this tag proving the emitted helper implementation. -/
  | lshr (t : PW)
  /- PTX definition: signed `shr` sign-fills and clamps excessive amounts. -/
  /- Lean modeling: the selected helper encodes GX's arithmetic-shift policy, whose ISO C implementation contract is external to this tag. -/
  | ashr (t : PW)
  /- PTX definition: unsigned `div` returns a quotient and leaves division-by-zero behavior unspecified. -/
  /- Lean modeling: the helper shape includes GX's all-ones zero-divisor policy; the ISA does not require that policy. -/
  | udiv (t : PW)
  /- PTX definition: signed `div` computes `d = a / b`, while division by zero yields an unspecified, machine-specific value. -/
  /- Lean modeling: truncation toward zero and the zero/`MIN ÷ -1` guards are GX/C helper policies, not additional results mandated by PTX. -/
  | sdiv (t : PW)
  /- PTX definition: unsigned `rem` returns the division remainder and leaves zero-divisor behavior unspecified. -/
  /- Lean modeling: the helper shape includes GX's dividend-on-zero policy; the ISA does not require that policy. -/
  | urem (t : PW)
  /- PTX definition: negative signed operands are machine-dependent, and division by zero yields an unspecified, machine-specific value. -/
  /- Lean modeling: the helper shape records GX's zero and minus-one guards, not an independent ISO C semantics. -/
  | srem (t : PW)
  /- PTX definition: unsigned `min` selects the lesser operand. -/
  /- Lean modeling: an unsigned cast supplies the intended order; equal operands make the branch tie choice observationally irrelevant. -/
  | umin (t : PW)
  /- PTX definition: unsigned `max` selects the greater operand. -/
  /- Lean modeling: an unsigned cast supplies the intended order; equal operands make the branch tie choice observationally irrelevant. -/
  | umax (t : PW)
  /- PTX definition: signed `min` selects the lesser two's-complement operand. -/
  /- Lean modeling: a signed cast supplies the intended order, but `t` is not formally linked here to Lean's value width. -/
  | smin (t : PW)
  /- PTX definition: signed `max` selects the greater two's-complement operand. -/
  /- Lean modeling: a signed cast supplies the intended order, but `t` is not formally linked here to Lean's value width. -/
  | smax (t : PW)
  /- PTX definition: `selp` selects `a` for true `p` and `b` for false `p`. -/
  /- Lean modeling: the C condition uses GX's zero/nonzero predicate representation; this tag does not model operand side effects. -/
  | sel
  /- PTX definition: integer `setp` applies a relation using instruction-directed signedness. -/
  /- Lean modeling: `sgn`, `c`, and `t` determine the printed comparison shape, without this constructor certifying its surrounding assignment. -/
  | cmp (sgn : Bool) (c : CmpOp) (t : PW)
  deriving Repr, DecidableEq

/- PTX definition: these value rules cover the indexed operations, while division by zero and signed-remainder edge behavior are unspecified or machine-specific. -/
/- Lean modeling: `evalC` transcribes each indexed C shape over `BitVec w`; it currently ignores every `t : PW` field, so it does not itself prove `t.bits = w`. -/
def CTemplate.evalC {w : Nat} : CTemplate → BitVec w → BitVec w → BitVec w → BitVec w
  /- PTX definition: integer `add` retains the low destination-width sum bits. -/
  /- Lean modeling: `BitVec w` addition gives that value, but this shallow arm ignores `t` and is not an ISO C printer certificate. -/
  | .add _,  a, b, _ => a + b
  /- PTX definition: `add.f16x2` adds the two packed half lanes independently. -/
  /- Lean modeling: the C-template value arm shares the bit-exact packed model later connected to the reviewed helper by the common printer boundary. -/
  | .addF16x2, a, b, _ => Ptx.addF16x2At a b
  /- PTX definition: integer `sub` retains the low destination-width difference bits. -/
  /- Lean modeling: `BitVec w` subtraction gives that value, but this shallow arm ignores `t` and does not interpret printed C. -/
  | .sub _,  a, b, _ => a - b
  /- PTX definition: `mul.lo` retains the destination-width low product bits. -/
  /- Lean modeling: `BitVec w` multiplication is the intended value; connecting `t` and printed subword conversion remains `gap:subword-template-certificate`. -/
  | .mul _,  a, b, _ => a * b
  /- PTX definition: `mul.bf16x2` multiplies the two packed bfloat16 lanes independently. -/
  /- Lean modeling: the template value arm shares the exact packed special-value and RN-even model with the named PTX instruction. -/
  | .mulBF16x2, a, b, _ => Ptx.mulBF16x2At a b
  /- PTX definition: `mad.lo` adds `c` to the destination-width low product. -/
  /- Lean modeling: fixed-width multiplication and addition model the result bits, while the arm neither inspects `t` nor proves printed C. -/
  | .mad _,  a, b, c => a * b + c
  /- PTX definition: integer `neg` stores the destination-width negated value. -/
  /- Lean modeling: `BitVec w` negation gives the required bits; correspondence with the `t`-indexed unsigned C expression is separate. -/
  | .neg _,  a, _, _ => -a
  /- PTX definition: bit-type `not` computes `d = ~a`. -/
  /- Lean modeling: fixed-width complement is the intended value, but this arm ignores `t` and does not formalize C integer promotions. -/
  | .bnot _, a, _, _ => ~~~a
  /- PTX definition: bit-type `and` computes `d = a & b`. -/
  /- Lean modeling: `BitVec` conjunction matches the result bits; the separate printer must still supply the intended fixed-width casts. -/
  | .band _, a, b, _ => a &&& b
  /- PTX definition: bit-type `or` computes `d = a | b`. -/
  /- Lean modeling: `BitVec` disjunction matches the result bits; the separate printer must still supply the intended fixed-width casts. -/
  | .bor _,  a, b, _ => a ||| b
  /- PTX definition: bit-type `xor` computes `d = a ^ b`. -/
  /- Lean modeling: `BitVec` xor matches the result bits; the separate printer must still supply the intended fixed-width casts. -/
  | .bxor _, a, b, _ => a ^^^ b
  /- PTX definition: predicate `not` reverses the Boolean source value. -/
  /- Lean modeling: this is C `!` at the value level and assumes side-effect-free already-evaluated operands. -/
  | .lnot,   a, _, _ => if a = 0 then 1 else 0
  /- PTX definition: predicate `and` is true exactly when both inputs are true. -/
  /- Lean modeling: zero/nonzero tests reproduce C `&&` results, without modeling short-circuit evaluation or side effects. -/
  | .land,   a, b, _ => if a ≠ 0 ∧ b ≠ 0 then 1 else 0
  /- PTX definition: predicate `or` is true when either input is true. -/
  /- Lean modeling: zero/nonzero tests reproduce C `||` results, without modeling short-circuit evaluation or side effects. -/
  | .lor,    a, b, _ => if a ≠ 0 ∨ b ≠ 0 then 1 else 0
  /- PTX definition: predicate `xor` is true exactly when its inputs differ. -/
  /- Lean modeling: inequality of truth values gives the required 0/1 result; this is a shallow value rule rather than ISO C evaluation. -/
  | .lxor,   a, b, _ => if (a ≠ 0) ↔ (b ≠ 0) then 0 else 1
  /- PTX definition: `shl` zero-fills and clamps excessive shift amounts to the width. -/
  /- Lean modeling: the guard avoids out-of-range C shifts at the value level, while `t.bits = w` is not established by this arm. -/
  | .shl _,  a, b, _ => if w ≤ b.toNat then 0#w else a <<< b.toNat
  /- PTX definition: unsigned `shr` zero-fills and clamps excessive shift amounts. -/
  /- Lean modeling: the guard and logical shift give the intended value, while helper printing and `t.bits = w` remain outside this arm. -/
  | .lshr _, a, b, _ => if w ≤ b.toNat then 0#w else a >>> b.toNat
  /- PTX definition: signed `shr` sign-fills and clamps excessive shift amounts. -/
  /- Lean modeling: `sshiftRight` supplies the value policy; the emitted helper's ISO C behavior and zero-width corner are not proved here. -/
  | .ashr _, a, b, _ => a.sshiftRight (if w ≤ b.toNat then w - 1 else b.toNat)
  /- PTX definition: unsigned `div` computes quotient, with division by zero unspecified. -/
  /- Lean modeling: this shallow arm records GX's chosen zero-divisor policy and ignores `t`; it is not a requirement of PTX. -/
  | .udiv _, a, b, _ => if b = 0 then ~~~(0#w) else a / b
  /- PTX definition: unsigned `rem` computes remainder, with division by zero unspecified. -/
  /- Lean modeling: this shallow arm records GX's chosen zero-divisor policy and ignores `t`; it is not a requirement of PTX. -/
  | .urem _, a, b, _ => if b = 0 then a else a % b
  /- PTX definition: signed `div` computes `d = a / b`, while division by zero yields an unspecified, machine-specific value. -/
  /- Lean modeling: `toInt.tdiv` and the zero/`MIN ÷ -1` guards specify GX's chosen C-helper policy, not extra behavior mandated by PTX. -/
  | .sdiv _, a, b, _ =>
      if b = 0 then ~~~(0#w)
      else if b.toInt = -1 then 0#w - a
      else BitVec.ofInt w (a.toInt.tdiv b.toInt)
  /- PTX definition: negative signed operands are machine-dependent, and division by zero yields an unspecified, machine-specific value. -/
  /- Lean modeling: `toInt.tmod` plus the guards specifies GX's value policy, not an independent semantics of the printed C helper. -/
  | .srem _, a, b, _ =>
      if b = 0 then a
      else if b.toInt = -1 then 0#w
      else BitVec.ofInt w (a.toInt.tmod b.toInt)
  /- PTX definition: unsigned `min` stores the lesser operand. -/
  /- Lean modeling: strict unsigned comparison chooses an equal-valued operand on ties; this value rule ignores the printed cast index `t`. -/
  | .umin _, a, b, _ => if a < b then a else b            -- (uN)a < (uN)b ? a : b
  /- PTX definition: unsigned `max` stores the greater operand. -/
  /- Lean modeling: strict unsigned comparison chooses an equal-valued operand on ties; this value rule ignores the printed cast index `t`. -/
  | .umax _, a, b, _ => if b < a then a else b            -- (uN)a > (uN)b ? a : b
  /- PTX definition: signed `min` stores the lesser signed operand. -/
  /- Lean modeling: `toInt` supplies signed order and tie choice is irrelevant; this arm does not connect `t` to `w`. -/
  | .smin _, a, b, _ => if a.toInt < b.toInt then a else b -- (iN)a < (iN)b ? a : b
  /- PTX definition: signed `max` stores the greater signed operand. -/
  /- Lean modeling: `toInt` supplies signed order and tie choice is irrelevant; this arm does not connect `t` to `w`. -/
  | .smax _, a, b, _ => if b.toInt < a.toInt then a else b
  /- PTX definition: `selp` stores `a` when `p` is true and `b` otherwise. -/
  /- Lean modeling: GX's word-level truth convention gives the intended value, without modeling C operand evaluation or side effects. -/
  | .sel,    p, a, b => if p ≠ 0 then a else b
  /- PTX definition: integer `setp` compares operands using relation and type-directed signedness. -/
  /- Lean modeling: bits, `toInt`, and `toNat` give the value relation; `t` is ignored and printed-cast correspondence is separate. -/
  | .cmp sgn c _, a, b, _ =>
      -- ((ctN)a OP (ctN)b) yields the C int 0/1; the iN/uN cast decides
      -- whether OP compares the signed (toInt) or unsigned (toNat) values.
      if (match c with
          | .eq => decide (a = b)
          | .ne => decide (a ≠ b)
          | .lt => if sgn then decide (a.toInt < b.toInt) else decide (a.toNat < b.toNat)
          | .le => if sgn then decide (a.toInt ≤ b.toInt) else decide (a.toNat ≤ b.toNat)
          | .gt => if sgn then decide (b.toInt < a.toInt) else decide (b.toNat < a.toNat)
          | .ge => if sgn then decide (b.toInt ≤ a.toInt) else decide (b.toNat ≤ a.toNat))
      then 1 else 0

namespace CTemplate

/- PTX definition: `add.u16` stores the low 16 bits of the unsigned sum. -/
/- Lean modeling: `.w32` supplies the promoted C shape; only value-level equality is proved and truncation remains `gap:subword-template-certificate`. -/
def addU16 : CTemplate := .add .w32

/- PTX definition: `add.s16` stores the low 16 bits of the signed sum. -/
/- Lean modeling: `.w32` supplies the promoted C shape; only value-level equality is proved and truncation remains `gap:subword-template-certificate`. -/
def addS16 : CTemplate := .add .w32

/- PTX definition: `add.u32` adds two 32-bit operands and retains the 32-bit result. -/
/- Lean modeling: the named C shape selects unsigned 32-bit casts, which make C addition defined modulo `2^32`. -/
def addU32 : CTemplate := .add .w32

/- PTX definition: `add.s32` has the same low 32-bit sum pattern as `add.u32`. -/
/- Lean modeling: the unsigned 32-bit C shape preserves that pattern while avoiding undefined signed-overflow behavior. -/
def addS32 : CTemplate := .add .w32

/- PTX definition: `add.u64` adds two 64-bit operands and retains the 64-bit result. -/
/- Lean modeling: the named C shape selects unsigned 64-bit casts, which make C addition defined modulo `2^64`. -/
def addU64 : CTemplate := .add .w64

/- PTX definition: `add.s64` has the same low 64-bit sum pattern as `add.u64`. -/
/- Lean modeling: the unsigned 64-bit C shape preserves that pattern while avoiding undefined signed-overflow behavior. -/
def addS64 : CTemplate := .add .w64

/- PTX definition: `sub.u16` stores the low 16 bits of the unsigned difference. -/
/- Lean modeling: `.w32` supplies the promoted C shape; only value-level equality is proved and truncation remains `gap:subword-template-certificate`. -/
def subU16 : CTemplate := .sub .w32

/- PTX definition: `sub.s16` stores the low 16 bits of the signed difference. -/
/- Lean modeling: `.w32` supplies the promoted C shape; only value-level equality is proved and truncation remains `gap:subword-template-certificate`. -/
def subS16 : CTemplate := .sub .w32

/- PTX definition: `sub.u32` stores the low 32 bits of the unsigned difference. -/
/- Lean modeling: the unsigned 32-bit template performs defined modular subtraction with the exact PTX result bits. -/
def subU32 : CTemplate := .sub .w32

/- PTX definition: `sub.s32` stores the low 32 bits of the signed difference. -/
/- Lean modeling: the unsigned 32-bit template preserves the two's-complement bits while avoiding signed-overflow behavior. -/
def subS32 : CTemplate := .sub .w32

/- PTX definition: `sub.u64` stores the low 64 bits of the unsigned difference. -/
/- Lean modeling: the unsigned 64-bit template performs defined modular subtraction with the exact PTX result bits. -/
def subU64 : CTemplate := .sub .w64

/- PTX definition: `sub.s64` stores the low 64 bits of the signed difference. -/
/- Lean modeling: the unsigned 64-bit template preserves the two's-complement bits while avoiding signed-overflow behavior. -/
def subS64 : CTemplate := .sub .w64

/- PTX definition: `div.u16` computes an unsigned 16-bit quotient for a nonzero divisor. -/
/- Lean modeling: this alias names the shallow quotient value model; production C uses the separate width-exact `printDivU16` spelling. -/
def divU16 : CTemplate := .udiv .w32

/- PTX definition: `div.s16` computes a signed 16-bit quotient for a nonzero divisor. -/
/- Lean modeling: this alias names the shallow quotient value model; production C uses the separate width-exact `printDivS16` spelling. -/
def divS16 : CTemplate := .sdiv .w32

/- PTX definition: `div.u32` computes an unsigned 32-bit quotient for a nonzero divisor. -/
/- Lean modeling: the guarded unsigned 32-bit helper has the PTX quotient and GX's explicit unspecified-case policy. -/
def divU32 : CTemplate := .udiv .w32

/- PTX definition: `div.s32` computes a signed 32-bit quotient for a nonzero divisor. -/
/- Lean modeling: the guarded signed 32-bit helper truncates toward zero and preserves GX's unspecified-case policy. -/
def divS32 : CTemplate := .sdiv .w32

/- PTX definition: `div.u64` computes an unsigned 64-bit quotient for a nonzero divisor. -/
/- Lean modeling: the guarded unsigned 64-bit helper has the PTX quotient and GX's explicit unspecified-case policy. -/
def divU64 : CTemplate := .udiv .w64

/- PTX definition: `div.s64` computes a signed 64-bit quotient for a nonzero divisor. -/
/- Lean modeling: the guarded signed 64-bit helper truncates toward zero and preserves GX's unspecified-case policy. -/
def divS64 : CTemplate := .sdiv .w64

/- PTX definition: `rem.u16` computes the unsigned 16-bit remainder for a nonzero divisor. -/
/- Lean modeling: this alias names the shallow remainder value model; production C uses the separate width-exact `printRemU16` spelling. -/
def remU16 : CTemplate := .urem .w32

/- PTX definition: `rem.s16` computes the signed 16-bit remainder for a nonzero divisor. -/
/- Lean modeling: this alias names the shallow remainder value model; production C uses the separate width-exact `printRemS16` spelling. -/
def remS16 : CTemplate := .srem .w32

/- PTX definition: `rem.u32` computes the unsigned 32-bit remainder for a nonzero divisor. -/
/- Lean modeling: the guarded unsigned 32-bit helper has the PTX remainder and GX's explicit unspecified-case policy. -/
def remU32 : CTemplate := .urem .w32

/- PTX definition: `rem.s32` computes the signed 32-bit remainder for a nonzero divisor. -/
/- Lean modeling: the guarded signed 32-bit helper implements GX's selected C-style signed policy. -/
def remS32 : CTemplate := .srem .w32

/- PTX definition: `rem.u64` computes the unsigned 64-bit remainder for a nonzero divisor. -/
/- Lean modeling: the guarded unsigned 64-bit helper has the PTX remainder and GX's explicit unspecified-case policy. -/
def remU64 : CTemplate := .urem .w64

/- PTX definition: `rem.s64` computes the signed 64-bit remainder for a nonzero divisor. -/
/- Lean modeling: the guarded signed 64-bit helper implements GX's selected C-style signed policy. -/
def remS64 : CTemplate := .srem .w64

/- PTX definition: `neg.s16` stores the 16-bit two's-complement negation of its operand. -/
/- Lean modeling: `.w32` supplies the promoted C shape; only value-level equality is proved and truncation remains `gap:subword-template-certificate`. -/
def negS16 : CTemplate := .neg .w32

/- PTX definition: `neg.s32` stores the 32-bit two's-complement negation of its operand. -/
/- Lean modeling: the unsigned 32-bit template computes defined modular negation with the exact result bits. -/
def negS32 : CTemplate := .neg .w32

/- PTX definition: `neg.s64` stores the 64-bit two's-complement negation of its operand. -/
/- Lean modeling: the unsigned 64-bit template computes defined modular negation with the exact result bits. -/
def negS64 : CTemplate := .neg .w64

/- PTX definition: `min.u16` selects the lesser unsigned 16-bit operand. -/
/- Lean modeling: this alias names the shallow unsigned-order value model; production C uses the separate width-exact `printMinU16` spelling. -/
def minU16 : CTemplate := .umin .w32

/- PTX definition: `min.s16` selects the lesser signed 16-bit operand. -/
/- Lean modeling: this alias names the shallow signed-order value model; production C uses the separate width-exact `printMinS16` spelling. -/
def minS16 : CTemplate := .smin .w32

/- PTX definition: `min.u32` selects the lesser unsigned 32-bit operand. -/
/- Lean modeling: the unsigned 32-bit comparison-and-select template uses the exact PTX order. -/
def minU32 : CTemplate := .umin .w32

/- PTX definition: `min.s32` selects the lesser signed 32-bit operand. -/
/- Lean modeling: the signed 32-bit comparison-and-select template uses the exact two's-complement order. -/
def minS32 : CTemplate := .smin .w32

/- PTX definition: `min.u64` selects the lesser unsigned 64-bit operand. -/
/- Lean modeling: the unsigned 64-bit comparison-and-select template uses the exact PTX order. -/
def minU64 : CTemplate := .umin .w64

/- PTX definition: `min.s64` selects the lesser signed 64-bit operand. -/
/- Lean modeling: the signed 64-bit comparison-and-select template uses the exact two's-complement order. -/
def minS64 : CTemplate := .smin .w64

/- PTX definition: `max.u16` selects the greater unsigned 16-bit operand. -/
/- Lean modeling: this alias names the shallow unsigned-order value model; production C uses the separate width-exact `printMaxU16` spelling. -/
def maxU16 : CTemplate := .umax .w32

/- PTX definition: `max.s16` selects the greater signed 16-bit operand. -/
/- Lean modeling: this alias names the shallow signed-order value model; production C uses the separate width-exact `printMaxS16` spelling. -/
def maxS16 : CTemplate := .smax .w32

/- PTX definition: `max.u32` selects the greater unsigned 32-bit operand. -/
/- Lean modeling: the unsigned 32-bit comparison-and-select template uses the exact PTX order. -/
def maxU32 : CTemplate := .umax .w32

/- PTX definition: `max.s32` selects the greater signed 32-bit operand. -/
/- Lean modeling: the signed 32-bit comparison-and-select template uses the exact two's-complement order. -/
def maxS32 : CTemplate := .smax .w32

/- PTX definition: `max.u64` selects the greater unsigned 64-bit operand. -/
/- Lean modeling: the unsigned 64-bit comparison-and-select template uses the exact PTX order. -/
def maxU64 : CTemplate := .umax .w64

/- PTX definition: `max.s64` selects the greater signed 64-bit operand. -/
/- Lean modeling: the signed 64-bit comparison-and-select template uses the exact two's-complement order. -/
def maxS64 : CTemplate := .smax .w64

/- PTX definition: `mul.lo.u16` keeps the low 16 bits of an unsigned 16-bit product. -/
/- Lean modeling: `evalC` at value width 16 retains the correct low bits, but linking `.w32` printed ISO C through destination conversion is still `gap:subword-template-certificate`. -/
def mulLoU16 : CTemplate := .mul .w32

/- PTX definition: `mul.lo.s16` keeps the same low 16 product bits under signed interpretation. -/
/- Lean modeling: `evalC` at value width 16 retains the sign-independent low bits, but linking `.w32` printed ISO C through destination conversion is still `gap:subword-template-certificate`. -/
def mulLoS16 : CTemplate := .mul .w32

/- PTX definition: `mul.lo.u32` keeps bits 31 through 0 of the unsigned product. -/
/- Lean modeling: the unsigned 32-bit multiplication template is defined modulo `2^32` and therefore returns exactly those bits. -/
def mulLoU32 : CTemplate := .mul .w32

/- PTX definition: `mul.lo.s32` keeps bits 31 through 0 of the signed product. -/
/- Lean modeling: low two's-complement product bits are sign-independent, so unsigned 32-bit C multiplication is exact and avoids signed-overflow UB. -/
def mulLoS32 : CTemplate := .mul .w32

/- PTX definition: `mul.lo.u64` keeps bits 63 through 0 of the unsigned product. -/
/- Lean modeling: the unsigned 64-bit multiplication template is defined modulo `2^64` and therefore returns exactly those bits. -/
def mulLoU64 : CTemplate := .mul .w64

/- PTX definition: `mul.lo.s64` keeps bits 63 through 0 of the signed product. -/
/- Lean modeling: low two's-complement product bits are sign-independent, so unsigned 64-bit C multiplication is exact and avoids signed-overflow UB. -/
def mulLoS64 : CTemplate := .mul .w64

/- PTX definition: `mad.lo.u16` retains the low 16 bits of the product-plus-addend result. -/
/- Lean modeling: `evalC` at 16 bits has the exact value; the printed 32-bit expression relies on destination truncation tracked by `gap:subword-template-certificate`. -/
def madLoU16 : CTemplate := .mad .w32

/- PTX definition: `mad.lo.s16` retains the low 16 two's-complement product-plus-addend bits. -/
/- Lean modeling: unsigned modular C arithmetic preserves those low bits; the printed subword assignment boundary remains separately tracked. -/
def madLoS16 : CTemplate := .mad .w32

/- PTX definition: `mad.lo.u32` retains the low 32 bits of the product plus addend. -/
/- Lean modeling: unsigned 32-bit multiplication and addition are defined modulo `2^32`, exactly matching the PTX result bits. -/
def madLoU32 : CTemplate := .mad .w32

/- PTX definition: `mad.lo.s32` retains the low 32 two's-complement product-plus-addend bits. -/
/- Lean modeling: unsigned modular arithmetic has the same low result pattern and avoids signed-overflow undefined behavior. -/
def madLoS32 : CTemplate := .mad .w32

/- PTX definition: `mad.lo.u64` retains the low 64 bits of the product plus addend. -/
/- Lean modeling: unsigned 64-bit multiplication and addition are defined modulo `2^64`, exactly matching the PTX result bits. -/
def madLoU64 : CTemplate := .mad .w64

/- PTX definition: `mad.lo.s64` retains the low 64 two's-complement product-plus-addend bits. -/
/- Lean modeling: unsigned modular arithmetic has the same low result pattern and avoids signed-overflow undefined behavior. -/
def madLoS64 : CTemplate := .mad .w64

/- PTX definition: every currently modeled scalar operation has one of the finite shapes indexed above. -/
/- Lean modeling: this explicit 71-entry inventory exposes both widths, packed binary16 add, packed bfloat16 multiply, every predicate shape, and all signedness/comparison combinations for mechanical completeness checks. -/
def all : List CTemplate := [
  .add .w32, .add .w64, .addF16x2, .sub .w32, .sub .w64,
  .mul .w32, .mul .w64, .mulBF16x2, .mad .w32, .mad .w64,
  .neg .w32, .neg .w64,
  .bnot .w32, .bnot .w64, .band .w32, .band .w64,
  .bor .w32, .bor .w64, .bxor .w32, .bxor .w64,
  .lnot, .land, .lor, .lxor,
  .shl .w32, .shl .w64, .lshr .w32, .lshr .w64,
  .ashr .w32, .ashr .w64,
  .udiv .w32, .udiv .w64, .sdiv .w32, .sdiv .w64,
  .urem .w32, .urem .w64, .srem .w32, .srem .w64,
  .umin .w32, .umin .w64, .umax .w32, .umax .w64,
  .smin .w32, .smin .w64, .smax .w32, .smax .w64,
  .sel,
  .cmp false .eq .w32, .cmp false .eq .w64,
  .cmp false .ne .w32, .cmp false .ne .w64,
  .cmp false .lt .w32, .cmp false .lt .w64,
  .cmp false .le .w32, .cmp false .le .w64,
  .cmp false .gt .w32, .cmp false .gt .w64,
  .cmp false .ge .w32, .cmp false .ge .w64,
  .cmp true .eq .w32, .cmp true .eq .w64,
  .cmp true .ne .w32, .cmp true .ne .w64,
  .cmp true .lt .w32, .cmp true .lt .w64,
  .cmp true .le .w32, .cmp true .le .w64,
  .cmp true .gt .w32, .cmp true .gt .w64,
  .cmp true .ge .w32, .cmp true .ge .w64
]

end CTemplate

end Gxptx
