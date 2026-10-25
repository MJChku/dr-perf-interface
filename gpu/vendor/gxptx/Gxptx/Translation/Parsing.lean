/-
  Gxptx.Translation.Parsing — trusted PTX token, type, grouping, and semantic
  selector parsing helpers. Generated-C operand/address/operator rendering is
  owned separately by Gxptx.Translation.C.Operands.
-/
import Gxptx.Semantics.Types
import Gxptx.Translation.C.Types
set_option linter.deprecated false
namespace Gxptx

/- PTX definition: PTX is line oriented and whitespace separates tokens without changing an instruction's operation. -/
/- Lean modeling: delegating to `String.trim` removes only boundary whitespace and leaves every interior PTX token byte unchanged. -/
@[inline] def strim (s : String) : String := s.trim

/- PTX definition: fundamental types include predicate, bit, signed/unsigned integer, and floating tokens at fixed widths. -/
/- Lean modeling: the parser preserves the established family routing; 8/16-bit integer forms use 32-bit value templates and therefore remain qualified by `gap:subword-template-certificate`. -/
def parseTy? (s : String) : Option Ty :=
  match s with
  | "pred" => some .pred
  | "b8" | "b16" | "b32" => some .b32
  | "u8" | "u16" | "u32" => some .u32
  | "s8" | "s16" | "s32" => some .s32
  | "b64" => some .b64
  | "u64" => some .u64
  | "s64" => some .s64
  | "f16" | "bf16" | "f32" | "f64" => some .f32
  | _ => none

/- PTX definition: instruction operands are comma-separated, while vector and call-tuple syntax encloses internal commas. -/
/- Lean modeling: a single nesting counter retains grouped substrings verbatim and splits only at depth zero, matching the shapes consumed by statement translation. -/
def splitArgs (s : String) : List String := Id.run do
  let mut out : Array String := #[]
  let mut cur : String := ""
  let mut depth : Nat := 0
  for c in s.toList do
    if c == '{' || c == '(' then depth := depth + 1; cur := cur.push c
    else if c == '}' || c == ')' then depth := depth - 1; cur := cur.push c
    else if c == ',' && depth == 0 then out := out.push cur; cur := ""
    else cur := cur.push c
  if ! (strim cur).isEmpty then out := out.push cur
  return (out.toList.map strim).filter (! ·.isEmpty)

/- PTX definition: memory operands are bracketed and may contain a comma-separated vector (`[tensorMap, {c0, c1}]`). -/
/- Lean modeling: splitting only at commas outside brackets, braces and parentheses keeps each bracketed operand whole; braces stripped by statement normalization do not change the grouping. -/
def splitOperands (s : String) : List String := Id.run do
  let mut out : Array String := #[]
  let mut cur : String := ""
  let mut depth : Nat := 0
  for c in s.toList do
    if c == '{' || c == '(' || c == '[' then depth := depth + 1; cur := cur.push c
    else if c == '}' || c == ')' || c == ']' then depth := depth - 1; cur := cur.push c
    else if c == ',' && depth == 0 then out := out.push cur; cur := ""
    else cur := cur.push c
  if ! (strim cur).isEmpty then out := out.push cur
  return (out.toList.map strim).filter (! ·.isEmpty)

/- PTX definition: a bracketed operand group lists an address followed by optional vector elements. -/
/- Lean modeling: removing brackets/braces and splitting at commas recovers the group's elements in order. -/
def bracketElements (s : String) : List String :=
  let inner := ((((strim s).replace "[" "").replace "]" "").replace "{" "").replace "}" ""
  ((inner.splitOn ",").map strim).filter (! ·.isEmpty)

/- PTX definition: call inputs are written as a parenthesized, comma-separated parameter tuple. -/
/- Lean modeling: removing only delimiters and the statement terminator before `splitArgs` preserves parameter order; malformed groups yield no fabricated arguments and are handled by the dispatcher. -/
def callInputParams (group : String) : List String :=
  let g := strim (group.replace ";" "")
  if g.startsWith "(" && g.endsWith ")" then
    splitArgs (strim ((g.drop 1).dropRight 1))
  else []

/- PTX definition: floating types/literals and typed registers distinguish floating values from integer bit patterns. -/
/- Lean modeling: the recognized `%f`/`0f`/`0d` prefixes match nvcc's emitted operand conventions and drive the existing bit-move-versus-assignment choice. -/
def isFloatOp (s : String) : Bool :=
  s.startsWith "%f" || s.startsWith "0f" || s.startsWith "0d"

/- PTX definition: registers are typed by declaration and scalar widths determine vector pack/unpack component sizes. -/
/- Lean modeling: testing the longest nvcc prefixes first maps its generated naming convention to 8/4/2/1-byte classes without guessing widths for non-register operands. -/
def regBytes? (s : String) : Option Nat :=
  if ! s.startsWith "%" then none
  else if s.startsWith "%rd" || s.startsWith "%fd" then some 8
  else if s.startsWith "%rs" || s.startsWith "%h" then some 2
  else if s.startsWith "%r" || s.startsWith "%f" then some 4
  else if s.startsWith "%p" then some 1
  else none

/- PTX definition: integer `cvt` forms interpret source and destination using signed/unsigned 8-, 16-, 32-, or 64-bit types. -/
/- Lean modeling: the closed mapping feeds exactly the cast dimensions used by the shallow conversion value model and rejects float/predicate tokens for separate handling. -/
def cvtIntTy? : String → Option (Bool × IW)
  | "s8"  => some (true, .w8)
  | "u8"  | "b8"  => some (false, .w8)
  | "s16" => some (true, .w16)
  | "u16" | "b16" => some (false, .w16)
  | "s32" => some (true, .w32)
  | "u32" | "b32" => some (false, .w32)
  | "s64" => some (true, .w64)
  | "u64" | "b64" => some (false, .w64)
  | _ => none

/- PTX definition: scalar floating instruction tokens include f16/f32/f64, while bf16 is an alternate format carried in b16 storage. -/
/- Lean modeling: the exact-token test recognizes scalar conversion/storage formats while excluding packed forms from scalar routes. -/
def isFloatTok (s : String) : Bool :=
  s == "f16" || s == "bf16" || s == "f32" || s == "f64"

/- PTX definition: supported scalar integer arithmetic uses only signed or unsigned 16-, 32-, and 64-bit type suffixes. -/
/- Lean modeling: this closed parser rejects packed, bit, floating, and modifier tokens so unsupported forms cannot silently reuse a scalar template. -/
def scalarArithTy? : String → Option (Bool × IW)
  | "u16" => some (false, .w16)
  | "s16" => some (true,  .w16)
  | "u32" => some (false, .w32)
  | "s32" => some (true,  .w32)
  | "u64" => some (false, .w64)
  | "s64" => some (true,  .w64)
  | _ => none

/- PTX definition: load/store element types determine scalar width, signedness, representation, and vector stride. -/
/- Lean modeling: one `MemTy` value selects both C access type and byte count, preventing width divergence; f16/bf16 deliberately move their 16-bit encodings. -/
def memTyOf : String → MemTy
  | "s8" => .s8 | "b8" | "u8" => .u8
  | "s16" => .s16 | "b16" | "u16" | "f16" | "bf16" => .u16
  | "s32" => .s32 | "b32" | "u32" => .u32
  | "s64" => .s64 | "b64" | "u64" => .u64
  | "f32" => .f32 | "f64" => .f64
  | _ => .u32

/- PTX definition: integer comparisons include eq/ne and signed or unsigned lt/le/gt/ge aliases. -/
/- Lean modeling: equivalent unsigned aliases share one value-level `CmpOp`; float-only unordered forms return `none` and remain on the explicitly trusted legacy printer path. -/
def cmpOp? : String → Option CmpOp
  | "eq" => some .eq | "ne" => some .ne
  | "lt" | "lo" => some .lt
  | "le" | "ls" => some .le
  | "gt" | "hi" => some .gt
  | "ge" | "hs" => some .ge
  | _ => none

end Gxptx
