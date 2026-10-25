/-
  Gxptx.Translation.Module.Registers — PTX register, parameter, and local
  declarations translated into the typed C locals used by a function body.
-/
import Gxptx.Translation.Parsing
import Gxptx.Translation.Body
import Gxptx.Translation.C.Operands
import Std.Data.HashSet
import Std.Data.HashMap
set_option linter.deprecated false
namespace Gxptx

private def paramIdentStart (c : UInt8) : Bool :=
  (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c == 95 || c == 36 || c == 37

private def paramIdentByte (c : UInt8) : Bool :=
  paramIdentStart c || (c >= 48 && c <= 57) || c == 46

-- Retain punctuation/whitespace so braces in instructions (vector operands)
-- are balanced too. Quoted strings are opaque, never rewritten identifiers.
private def paramTokens (src : String) : Array String := Id.run do
  let bs := src.toUTF8
  let mut out := #[]
  let mut i := 0
  while i < bs.size do
    let start := i
    if bs[i]! == 34 then
      i := i + 1
      while i < bs.size && bs[i]! != 34 do
        if bs[i]! == 92 && i + 1 < bs.size then i := i + 1
        i := i + 1
      if i < bs.size then i := i + 1
    else if paramIdentByte bs[i]! then
      i := i + 1
      while i < bs.size && paramIdentByte bs[i]! do i := i + 1
    else
      i := i + 1
    out := out.push (String.fromUTF8! (bs.extract start i))
  return out

/- PTX definition: call-parameter objects belong to their lexical brace scope;
   nvcc reuses names such as param0 for unrelated types in different scopes. -/
/- Lean modeling: alpha-renaming each body-local parameter preserves its uses
   and shadowing before C declaration hoisting. Size/alignment stay unchanged;
   this is ABI/name-resolution plumbing, not a new timing behavior. -/
def scopeCallParams (src : String) : String := Id.run do
  let tokens := paramTokens (stripLineDirectives (stripComments src))
  let mut used : Std.HashSet String := {}
  for t in tokens do used := used.insert t
  let mut scopes : Array (Std.HashMap String String) := #[{}]
  let mut next := 0
  let mut out : Array String := #[]
  for i in [0:tokens.size] do
    let t := tokens[i]!
    if t == "{" then scopes := scopes.push {}
    if t == "}" && scopes.size > 1 then scopes := scopes.pop
    if t == ".param" then
      let mut j := i + 1
      -- Types and qualifiers start with '.', extents/alignment with a digit;
      -- the first identifier is the declared scalar or array name.
      while j < tokens.size && tokens[j]! != ";" &&
          !(paramIdentStart (tokens[j]!.toUTF8[0]!)) do
        j := j + 1
      if j < tokens.size && tokens[j]! != ";" then
        let name := tokens[j]!
        let mut fresh := s!"gxCallArg{next}"
        while used.contains fresh do
          next := next + 1
          fresh := s!"gxCallArg{next}"
        next := next + 1
        used := used.insert fresh
        let k := scopes.size - 1
        scopes := scopes.set! k (scopes[k]!.insert name fresh)
    let mut renamed := t
    for scope in scopes.toList.reverse do
      if let some name := scope[t]? then
        renamed := name
        break
    out := out.push renamed
  return String.join out.toList

/- PTX definition: a `.reg` declared inside a `{ }` block is local to that
   block; separate inline-assembly blocks reuse names such as `tmp` with
   different types (`.b16` in one, `.b32` in another). -/
/- Lean modeling: alpha-renaming each plain-identifier register declaration to
   a fresh name within its lexical scope preserves every use and shadowing
   before C hoists all declarations to function scope; `%`-prefixed and
   `name<N>` register families are left unchanged. -/
def scopeRegs (src : String) : String := Id.run do
  let tokens := paramTokens src
  let mut used : Std.HashSet String := {}
  for t in tokens do used := used.insert t
  let mut scopes : Array (Std.HashMap String String) := #[{}]
  let mut next := 0
  let mut out : Array String := #[]
  let mut i := 0
  while i < tokens.size do
    let t := tokens[i]!
    if t == "{" then scopes := scopes.push {}
    if t == "}" && scopes.size > 1 then scopes := scopes.pop
    if t == ".reg" then
      -- Declared names: identifier tokens up to `;` that are not type or
      -- qualifier words (leading '.'), not `%` families and not `name<N>`.
      let mut j := i + 1
      while j < tokens.size && tokens[j]! != ";" do
        let x := tokens[j]!
        let mut k := j + 1
        while k < tokens.size && (strim tokens[k]!).isEmpty do k := k + 1
        if !x.isEmpty && paramIdentStart x.toUTF8[0]! && !x.startsWith "." && !x.startsWith "%" &&
            tokens[k]?.getD "" != "<" then
          let mut fresh := s!"gxScopedReg{next}_{x}"
          while used.contains fresh do
            next := next + 1
            fresh := s!"gxScopedReg{next}_{x}"
          next := next + 1
          used := used.insert fresh
          let top := scopes.size - 1
          scopes := scopes.set! top (scopes[top]!.insert x fresh)
        j := j + 1
    let mut renamed := t
    for scope in scopes.toList.reverse do
      if let some name := scope[t]? then
        renamed := name
        break
    out := out.push renamed
    i := i + 1
  return String.join out.toList

/- PTX definition: branch labels in nested scopes can be reused by separate
   inline-assembly statements; a forward branch resolves in its lexical scope. -/
/- Lean modeling: a two-pass alpha-renaming preserves all branch edges before
   C flattens the scopes. No wait-loop labels or branches may be discarded. -/
def scopeLabels (src : String) : String := Id.run do
  let tokens := paramTokens (stripLineDirectives (stripComments src))
  let mut maps : Array (Std.HashMap String String) := #[{}]
  let mut stack : Array Nat := #[0]
  let mut used : Std.HashSet String := {}
  for t in tokens do used := used.insert t
  let mut freshId := 0
  for i in [0:tokens.size] do
    let t := tokens[i]!
    if t == "{" then
      stack := stack.push maps.size
      maps := maps.push {}
    if t == "}" && stack.size > 1 then stack := stack.pop
    let mut j := i + 1
    while j < tokens.size && (strim tokens[j]!).isEmpty do j := j + 1
    if !t.isEmpty && paramIdentStart t.toUTF8[0]! &&
        j < tokens.size && tokens[j]! == ":" &&
        tokens[j+1]?.getD "" != ":" && tokens[i-1]?.getD "" != ":" then
      let mut fresh := s!"gxScopedLabel{freshId}"
      while used.contains fresh do
        freshId := freshId + 1
        fresh := s!"gxScopedLabel{freshId}"
      freshId := freshId + 1
      used := used.insert fresh
      let k := stack.back!
      maps := maps.set! k (maps[k]!.insert t fresh)
  stack := #[0]
  let mut nextScope := 1
  let mut out : Array String := #[]
  for t in tokens do
    if t == "{" then
      stack := stack.push nextScope
      nextScope := nextScope + 1
    if t == "}" && stack.size > 1 then stack := stack.pop
    let mut renamed := t
    for k in stack.toList.reverse do
      if let some name := maps[k]![t]? then
        renamed := name
        break
    out := out.push renamed
  return String.join out.toList

/- PTX definition: registers and local/parameter variables have declared types, extents, and alignments. -/
/- Lean modeling: the scanner preserves declared widths/counts and emits addressable buffers for `.param`/`.local` storage; naming, zero-initialization, and host alignment are trusted translation/ABI choices. -/
def regLocals (src : String) : String := Id.run do
  let mut out : Array String := #[]
  let mut seen : Std.HashSet String := {}
  -- scan STATEMENTS (`.reg`/`.param` may sit mid-line inside `{ … }` blocks).
  -- Strip `//` comments FIRST (else flattening newlines merges a comment line
  -- with a following `.reg` decl, hiding it) — the same trap as translateBody.
  let noc := stripLineDirectives (stripComments src)
  let flat := ((((noc.replace "{" " ").replace "}" " ").replace "\t" " ")).replace "\n" " "
  -- Normalize only register name lists; param/local arrays and alignment keep
  -- the established scanner/ABI path. Use exactly the same declaration parser
  -- as bare-register operand recognition.
  let stmts := (flat.splitOn ";").flatMap (fun stmt =>
    match registerDeclaration? stmt with
    | some (ty, names) => names.map (fun name => s!".reg {ty} {name}")
    | none => [stmt])
  for stmt in stmts do
    -- Labels have no semicolon, so a following declaration shares this chunk.
    let words := ((strim stmt).splitOn " ").filter (! ·.isEmpty)
    let l := String.intercalate " " (words.dropWhile (·.endsWith ":"))
    if l.startsWith ".reg" || l.startsWith ".param" || l.startsWith ".local" then
      let isParam := l.startsWith ".param" || l.startsWith ".local"
      let toks := (l.splitOn " ").filter (! ·.isEmpty)
      -- last token holds the name/spec; a `.param .align A .b8 x[N]` has the
      -- name last too
      match toks.reverse with
      | last :: tyRest =>
        let hadPct := last.contains '%'
        let spec := ((last.replace ";" "").replace "%" "")
        -- C type: the `.bN/.uN/...` token (second-to-last for arrays)
        let tyTok := (tyRest.filterMap (fun t =>
          let t := t.replace "." ""
          if ["b8","b16","b32","b64","b128","u8","u16","u32","u64","s8","s16","s32","s64","f16","f32","f64","pred"].contains t
          then some t else none)).getLastD "b32"
        let ct := ctypeOf tyTok
        let requestedAlign := match toks.dropWhile (· != ".align") with
          | _ :: a :: _ => a.toNat?.getD 0
          | _ => 0
        -- array form  name[N]
        match (spec.replace "]" "").splitOn "[" with
        | [nm, szs] =>
          if ! seen.contains nm then
            seen := seen.insert nm
            let declPrefix := if requestedAlign > 1 then s!"alignas({requestedAlign}) " else ""
            out := out.push (s!"    {declPrefix}{ct} {nm}[{szs}] = " ++ "{0};\n")
        | _ =>
          match (spec.replace ">" "").splitOn "<" with
          | [pfx, cnt] =>
            match cnt.toNat? with
            | some n => for i in [0:n] do
                let cname := if hadPct then s!"R_{pfx}{i}" else s!"{pfx}{i}"
                if !seen.contains cname then
                  seen := seen.insert cname
                  out := out.push s!"    {ct} {cname} = 0;\n"
            | none => pure ()
          | _ =>
            -- `.param`/`.local` scalars are call-marshalling BUFFERS addressed by
            -- `[name]` (st.param/ld.param), NOT value registers: emit a byte array
            -- so `(uintptr_t)name` yields the buffer address (Python's `name_base`).
            if isParam && (spec.splitOn "_param_").length == 1 then
              if ! seen.contains spec then
                seen := seen.insert spec
                let sz := if tyTok.endsWith "8" then 1 else if tyTok.endsWith "16" then 2
                          else if tyTok.endsWith "64" then 8 else 4
                let al := if requestedAlign == 0 then sz else requestedAlign
                -- zero-init: a branch-dependent ld.param before any st.param
                -- reads garbage otherwise — UB the optimizer exploits (-O1
                -- miscompiled SendRecv's warp partition; fine at -O0).
                out := out.push (s!"    alignas({al}) unsigned char {spec}[{sz}] = " ++ "{0};\n")
            else
              let cname := if hadPct && ! isParam then s!"R_{spec}" else spec
              if ! seen.contains cname then
                seen := seen.insert cname
                out := out.push s!"    {ct} {cname} = 0;\n"
      | _ => pure ()
  return String.join out.toList

end Gxptx
