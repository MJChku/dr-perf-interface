/-
  Gxptx.Translation.C.Operands — PTX operand and destination spellings at the
  generated-C boundary. Parsing remains string based; this module owns only
  the conversion from parsed operand text to C expression/lvalue text.
-/
import Gxptx.Translation.C.Expression
import Std.Data.HashSet
set_option linter.deprecated false
namespace Gxptx

/- PTX definition: special registers expose lane, thread/block, clock, environment, and timer state through `%`-prefixed names. -/
/- Lean modeling: supported names select runtime accessors, including linear-thread-derived lane/warp ids; envreg1/2 expose the per-launch cooperative-grid workspace and all other names remain `gap:special-register-coverage`. -/
def sreg? (s : String) : Option String :=
  let vec (nm : String) : Option String :=
    match (s.splitOn ".") with
    | [base, lane] => if base == nm then some ("_gx_" ++ nm.drop 1 ++ "." ++ lane) else none
    | _ => none
  match s with
  | "%laneid"      => some "(gx_lane_id())"
  | "%warpid"      => some "(gx_warp_id())"
  | "%clock"       => some "((uint32_t)gx_clock64())"
  | "%clock64" | "%globaltimer" => some "(gx_clock64())"
  -- CUDA cooperative groups loads the driver-provided grid workspace address
  -- from envreg1/2. GX owns an equivalent workspace for every CPU launch.
  | "%envreg1" => some "(gx_dev_envreg(1))"
  | "%envreg2" => some "(gx_dev_envreg(2))"
  | "%cluster_ctarank" => some "(gx_ptx_cluster_ctarank())"
  | "%cluster_nctarank" => some "(gx_dev_cluster_nctarank())"
  | "%is_explicit_cluster" => some "(gx_dev_is_explicit_cluster())"
  | "%clusterid.x" => some "(gx_ptx_cluster_id().x)"
  | "%clusterid.y" => some "(gx_ptx_cluster_id().y)"
  | "%clusterid.z" => some "(gx_ptx_cluster_id().z)"
  | "%nclusterid.x" => some "(gx_ptx_cluster_grid().x)"
  | "%nclusterid.y" => some "(gx_ptx_cluster_grid().y)"
  | "%nclusterid.z" => some "(gx_ptx_cluster_grid().z)"
  | _ =>
    let clusterVec (nm fn : String) : Option String :=
      match s.splitOn "." with
      | [base, lane] => if base == nm && ["x", "y", "z"].contains lane
                        then some s!"({fn}().{lane})" else none
      | _ => none
    (vec "%tid").orElse fun _ => (vec "%ntid").orElse fun _ =>
    (vec "%ctaid").orElse fun _ => (vec "%nctaid").orElse fun _ =>
    (clusterVec "%cluster_ctaid" "gx_dev_cluster_ctaid").orElse fun _ =>
    (clusterVec "%cluster_nctaid" "gx_dev_cluster_nctaid")

/- PTX definition: instruction operands may be registers, immediates, symbols, special registers, or vector groups. -/
/- Lean modeling: the recursive cases preserve the historical C spelling for every supported operand class; literal width interpretation is deferred to the instruction context instead of being prematurely truncated. -/
partial def operand (s : String) : CExpr :=
  let s := s.trim
  if s.startsWith "{" then
    let inner := ((s.drop 1).replace "}" "").splitOn ","
    operand ((inner.headD "0").trim)
  else
  match sreg? s with
  | some e => .var e
  | none =>
    if (s.splitOn "_param_").length > 1 then
      -- by-value kernel param → args[idx]
      let idx := ((s.splitOn "_param_").getLastD "0").takeWhile Char.isDigit
      .var s!"((uint64_t)(uintptr_t)args[{idx}])"
    else if s.startsWith "%" then .var ("R_" ++ s.drop 1)
    else match (s.toInt?) with
      -- Integer literals must NOT be pre-truncated to a fixed width: the same
      -- constant appears in both `.b32` and `.b64` contexts, and C's usual
      -- arithmetic conversions widen it to whichever operand it meets. A small
      -- non-negative value keeps the verified 32-bit `.lit`; a negative one is
      -- emitted as a SIGNED C literal so `x & -4096` sign-extends to the op
      -- width (a 64-bit `& 0xFFFFFFFFFFFFF000`), not a truncated `0xFFFFF000`;
      -- a large non-negative one gets a 64-bit `ull` literal.
      | some i =>
        if 0 ≤ i && i < 4294967296 then .lit (BitVec.ofNat 32 i.toNat)
        else if i < 0 then .var s!"({i})"
        else .var s!"{i}ull"
      | none   =>
          -- float bit literal `0f3F800000` → reinterpret bits
          if s.startsWith "0f" || s.startsWith "0F" then .var s!"gx_bits_to_f32(0x{s.drop 2}u)"
          else if s.startsWith "0d" || s.startsWith "0D" then .var s!"gx_bits_to_f64(0x{s.drop 2}ull)"
          -- a bare symbol used as a value: `(uintptr_t)` yields the ADDRESS for
          -- arrays (globals, tables, local depots) and the VALUE for scalars —
          -- exactly right for both, so cast uniformly.
          else .var s!"((uint64_t)(uintptr_t){s})"

/- PTX definition: register operands are `%`-prefixed and result instructions write their first destination operand. -/
/- Lean modeling: removing `%` and adding the collision-safe `R_` prefix exactly matches GX local-register declarations, while non-register destinations remain unchanged. -/
def dest (s : String) : String :=
  let s := s.trim
  if s.startsWith "%" then "R_" ++ s.drop 1 else s

/- PTX definition: fundamental PTX types distinguish bit width, signedness, predicate, and floating representation. -/
/- Lean modeling: each supported token maps to a same-width C access/conversion type; f16/bf16 remain 16-bit patterns and the unknown-token fallback preserves legacy behavior. -/
def ctypeOf : String → String
  | "b8"  | "u8"  => "uint8_t"   | "s8"  => "int8_t"
  | "b16" | "u16" | "f16" | "bf16" => "uint16_t"  | "s16" => "int16_t"
  | "b32" | "u32" => "uint32_t"  | "s32" => "int32_t"
  | "b64" | "u64" => "uint64_t"  | "s64" => "int64_t"
  | "b128" => "unsigned __int128"
  | "f32" => "float" | "f64" => "double" | "pred" => "int"
  | _ => "uint32_t"

/- PTX definition: source operands supply the values consumed by an instruction's operation rule. -/
/- Lean modeling: composing `operand` with `CExpr.toC` preserves the established parsing and printer routes; the final printer/compiler correspondence is explicitly trusted. -/
def oC (s : String) : String := (operand s).toC

/- PTX definition: declarations, not register spelling, determine storage; PTX ISA §9.4 allows bit-size registers to carry floating operands. -/
/- Lean modeling: bare declared registers are C value locals, not symbol addresses; actual C local types supply representations, with helper/compiler behavior trusted. -/
def registerOperand (regs : Std.HashSet String) (s : String) : CExpr :=
  if regs.contains s.trim && !(s.trim.startsWith "%") then .var s.trim
  else operand s

/- PTX definition: decimal floating constants have optional point/exponent and represent double values; exact 0f/0d hex encodings are a separate syntax (§4.5.2). -/
/- Lean modeling: this lexical recognizer prevents scalar decimal/exponent literals from being rendered as symbol addresses; general constant expressions remain outside this parser. -/
private def decimalFloatLiteral (s : String) : Bool :=
  let unsigned := if s.startsWith "-" || s.startsWith "+" then s.drop 1 else s
  let parts := (unsigned.replace "E" "e").splitOn "e"
  let digits (s : String) := !s.isEmpty && s.toList.all Char.isDigit
  let mantissa (s : String) :=
    match s.splitOn "." with
    | [whole] => digits whole
    | [whole, fraction] => digits (whole ++ fraction) &&
        whole.toList.all Char.isDigit && fraction.toList.all Char.isDigit
    | _ => false
  match parts with
  | [m] => m.contains '.' && mantissa m
  | [m, e] =>
    let exponent := if e.startsWith "-" || e.startsWith "+" then e.drop 1 else e
    mantissa m && digits exponent
  | _ => false

/- PTX definition: f32/f64 sources interpret register bits at instruction precision, and double-valued immediate constants are converted to that precision before use (§4.5.2). -/
/- Lean modeling: declared register representations use memcpy; scalar numeric immediates receive an explicit precision cast before arithmetic, with host conversion/IEEE correspondence trusted. -/
def floatOperand (regs : Std.HashSet String) (tok s : String) : String :=
  let s := s.trim
  let x := if decimalFloatLiteral s then s else (registerOperand regs s).toC
  if s.trim.startsWith "%" || regs.contains s.trim then
    s!"gx_as_f<{ctypeOf tok}>({x})"
  else if (tok == "f32" && (s.startsWith "0f" || s.startsWith "0F")) ||
      (tok == "f64" && (s.startsWith "0d" || s.startsWith "0D")) then x
  else s!"({ctypeOf tok})({x})"

/- PTX definition: b32/b64 operations consume raw encodings even for floating register sources or exact IEEE hex literals. -/
/- Lean modeling: width-selected memcpy supplies integer bits without numerical conversion; printing/helper/compiler correspondence remains trusted. -/
def bitOperand (regs : Std.HashSet String) (tok s : String) : String :=
  let x := (registerOperand regs s).toC
  if s.trim.startsWith "%" || regs.contains s.trim ||
      s.trim.startsWith "0f" || s.trim.startsWith "0F" ||
      s.trim.startsWith "0d" || s.trim.startsWith "0D" then
    s!"gx_f_as<{ctypeOf tok}>({x})"
  else s!"({ctypeOf tok})({x})"

/- PTX definition: floating comparison/minmax sources are IEEE encodings of values already converted to the instruction precision, not integer encodings of numeric immediates. -/
/- Lean modeling: register and matching exact-hex patterns retain raw bits; other immediates use floatOperand's precision conversion before bit extraction, with conversion/helper/compiler correspondence trusted. -/
def floatBitsOperand (regs : Std.HashSet String) (tok s : String) : String :=
  let bits := if tok == "f64" then "b64" else "b32"
  let s := s.trim
  if s.startsWith "%" || regs.contains s ||
      (tok == "f32" && (s.startsWith "0f" || s.startsWith "0F")) ||
      (tok == "f64" && (s.startsWith "0d" || s.startsWith "0D")) then
    bitOperand regs bits s
  else s!"gx_f_as<{ctypeOf bits}>({floatOperand regs tok s})"

/- PTX definition: floating results are rounded to instruction precision and their encodings are written into destination register storage. -/
/- Lean modeling: explicit float/double result casts fix width before memcpy, avoiding destination value conversion; host rounding, representation and compiler behavior remain trusted. -/
def writeFloat (tok d value : String) : String :=
  s!"    gx_movb({dest d}, ({ctypeOf tok})({value}));"

/- PTX definition: ordered floating comparisons reject NaN, unordered variants accept NaN, and num/nan test operand classification. -/
/- Lean modeling: a closed selector maps the ISA comparison names to the bit-based runtime helper; unsupported names fail closed rather than inheriting integer aliases. -/
def floatCmpCode? : String → Option Nat
  | "eq" => some 0 | "ne" => some 1 | "lt" => some 2
  | "le" => some 3 | "gt" => some 4 | "ge" => some 5
  | "equ" => some 6 | "neu" => some 7 | "ltu" => some 8
  | "leu" => some 9 | "gtu" => some 10 | "geu" => some 11
  | "num" => some 12 | "nan" => some 13 | _ => none

/- PTX definition: f32/f64 setp consumes same-width IEEE encodings; only f32 permits flushing subnormal operands to signed zero. -/
/- Lean modeling: operand bitcasts preserve declared register representations and constants; the tested encoding comparator is trusted, not covered by integer comparison theorems. -/
def printFloatSetp? (regs : Std.HashSet String) (cmp ty d a b : String)
    (ftz : Bool := false) : Option String := do
  if d == "_" || (d.splitOn "|").length != 1 ||
      !(ty == "f32" || ty == "f64") || (ftz && ty != "f32") then none
  else
    let code ← floatCmpCode? cmp
    return s!"    {dest d} = gx_setp_float_bits({floatBitsOperand regs ty a}, {floatBitsOperand regs ty b}, {code}u, {if ftz then "true" else "false"});"

/- PTX definition: memory operands use `[address]`, where an address may be a register or register/symbol plus a constant byte offset. -/
/- Lean modeling: stripping brackets and preserving the parsed base/offset order reproduces the existing C address expression; richer address forms remain outside this renderer. -/
def parseAddr (mem : String) (regs : Std.HashSet String := {}) : String :=
  let oC (s : String) := (registerOperand regs s).toC
  let inner := (mem.replace "[" "" |>.replace "]" "").trim
  match inner.splitOn "+" with
  | [b]    => oC b.trim
  | [b, o] => s!"({oC b.trim} + {o.trim})"
  | b :: _ => oC b.trim
  | []     => "0"

/- PTX definition: `setp` supports equality and ordered comparison operators, including unsigned aliases. -/
/- Lean modeling: aliases with the same ordering map to one C token; this renderer is retained for float/unordered forms and its printer/compiler semantics remain trusted. -/
def cmpC : String → Option String
  | "eq" => some "==" | "ne" => some "!="
  | "lt" | "ltu" | "lo" => some "<"
  | "le" | "leu" | "ls" => some "<="
  | "gt" | "gtu" | "hi" => some ">"
  | "ge" | "geu" | "hs" => some ">="
  | _ => none

end Gxptx
