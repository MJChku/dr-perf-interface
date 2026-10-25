/-
  Gxptx.Translation.Body — whole-body preprocessing and statement scheduling.
  It strips source-only directives, invokes translateStmt, and inserts the
  cooperative yield required on backward branches.
-/
import Gxptx.Translation.Statement
import Std.Data.HashSet
set_option linter.deprecated false
namespace Gxptx

/- PTX definition: line comments extend from `//` to the end of the source line and do not contribute instruction tokens. -/
/- Lean modeling: truncating each physical line at its first comment marker preserves preceding PTX text and prevents comment text from merging into the next statement. -/
def stripComments (src : String) : String :=
  String.intercalate "\n" ((src.splitOn "\n").map (fun line =>
    (line.splitOn "//").headD line))

/- PTX definition: `.pragma`, `.file`, and `.loc` are performance/debug directives rather than instruction statements. -/
/- Lean modeling: removing exactly those line prefixes prevents a non-semicolon directive from swallowing the following instruction while preserving every other source line. -/
def stripLineDirectives (src : String) : String :=
  String.intercalate "\n" ((src.splitOn "\n").map (fun line =>
    let l := strim line
    let head := (((l.replace "\t" " ").splitOn " ").filter (! ·.isEmpty)).headD ""
    if head == ".loc" || head == ".file" || head == ".pragma"
    then "" else line))

/- PTX definition: a `bra` instruction names the label to which execution transfers. -/
/- Lean modeling: recognizing the fixed `goto L_...;` spelling produced by `translateStmt` recovers only translator-generated branch edges for scheduler-yield insertion. -/
def gotoTarget (line : String) : Option String :=
  match (line.splitOn "goto ") with
  | _ :: t :: _ => some (strim ((t.splitOn ";").headD t))
  | _ => none

/- PTX definition: branch targets are labels defined in the containing function's instruction sequence. -/
/- Lean modeling: the fixed `L_...: ;` recognizer matches only translator-generated labels and leaves unrelated C text outside the control-flow scan. -/
def labelDef (line : String) : Option String :=
  let l := strim line
  if l.startsWith "L_" then some (strim ((l.splitOn ":").headD l)) else none

/- PTX definition: function bodies contain labeled instructions whose branches may form retry loops. -/
/- Lean modeling: statement order is preserved and only a goto to an already emitted label gains scheduler plumbing; parsing/control-flow and yield fairness remain trusted structural behavior. -/
private def payloadScopeStore? (stmt : String) : Option Bool :=
  let s := strim ((((stmt.replace "\t" " ").replace "\n" " ").replace "{" " ").replace "}" " ")
  let parts := s.splitOn " " |>.filter (! ·.isEmpty)
  match parts with
  | op :: _ =>
      if op != "st.param.b32" then none
      else
        let args := splitArgs (strim (s.drop op.length))
        match args with
        | _ :: value :: _ =>
            if strim value == "0" then some false
            else if strim value == "1" then some true
            else none
        | _ => none
  | _ => none

-- Register names can be arbitrary, including bare inline-assembly locals.
-- Keep the C declared storage type authoritative; this set only distinguishes
-- bare register values from bare symbol addresses in operand rendering.
/- PTX definition: .reg declarations associate one type with a comma-separated list of scalar names and numbered groups. -/
/- Lean modeling: shared name-list parsing keeps C declarations and bare-value recognition consistent across optional spaces, commas and group extents; full grammar remains trusted. -/
def registerDeclaration? (stmt : String) : Option (String × List String) :=
  let flat := (((stmt.replace "{" " ").replace "}" " ").replace "\t" " ").replace "\n" " "
  let words := (flat.splitOn " ").filter (! ·.isEmpty)
  match words.dropWhile (· != ".reg") with
  | ".reg" :: ty :: names => some (ty, splitArgs (String.intercalate " " names))
  | _ => none

/- PTX definition: .reg declarations name scalar values or numbered groups independently of optional percent/name conventions. -/
/- Lean modeling: this set separates bare C values from symbol addresses; declarations supply types while shadowing, full grammar and type legality remain trusted parsing boundaries. -/
def bodyRegisters (body : String) : Std.HashSet String := Id.run do
  let mut regs : Std.HashSet String := {}
  for stmt in body.splitOn ";" do
    match registerDeclaration? stmt with
    | some (ty, names) =>
      -- `name#pred` additionally marks predicate registers, for operands
      -- whose PTX meaning depends on the register type (cp.async ignore-src).
      let tag (x : String) := if ty == ".pred" then [x, x ++ "#pred"] else [x]
      for name in names do
        match (name.replace ">" "").splitOn "<" with
        | [base, count] =>
          if let some n := count.toNat? then
            for i in [0:n] do
              for x in tag s!"{base}{i}" do regs := regs.insert x
        | _ => for x in tag name do regs := regs.insert x
    | _ => pure ()
  return regs

def translateBodyScoped (body : String) (payloadAnnotated : Bool) : List String := Id.run do
  let source := stripLineDirectives (stripComments body)
  let stmts := source.splitOn ";"
  let regs := bodyRegisters source
  let mut seen : Std.HashSet String := {}
  let mut out : Array String := #[]
  -- Ordinary modules retain GX's runtime fake-address guard. An annotated
  -- NCCL module starts outside payload scope, so its real control memory is
  -- emitted as checked host loads/stores. Scope controls payload accounting,
  -- never whether fake-buffer safety checks are required. NVCC materializes scope(0/1) as an
  -- immediate st.param.b32 followed by the marker call; remember that value.
  let mut inPayload := !payloadAnnotated
  let mut pendingScope : Option Bool := none
  for st in stmts do
    if let some enabled := payloadScopeStore? st then
      pendingScope := some enabled
    let isScopeCall := (st.splitOn "gx_ptx_payload_scope").length > 1 &&
                       (st.splitOn "call").length > 1
    let line := translateStmtScoped st inPayload regs
    if line.isEmpty then continue
    match labelDef line with
    | some lbl => seen := seen.insert lbl
    | none => pure ()
    match gotoTarget line with
    | some tgt => if seen.contains tgt then out := out.push "    gx_ptx_spin_yield();"
    | none => pure ()
    out := out.push line
    if isScopeCall then
      -- Unknown marker arguments fall back to guarded accesses rather than
      -- risking a host dereference of fake device memory.
      inPayload := pendingScope.getD true
      pendingScope := none
  return out.toList

def translateBody (body : String) : List String :=
  translateBodyScoped body false

end Gxptx
