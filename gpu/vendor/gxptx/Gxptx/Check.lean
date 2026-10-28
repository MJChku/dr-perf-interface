/-
  Gxptx.Check — `lake exe gxcheck`: the verification-coverage checker.

  1. BUILD-TIME (elaborating this file): `#assert_axioms` fails the build
     unless every root certificate's proof closure uses only
     propext/Quot.sound — mechanically rejecting `sorry` (sorryAx) and
     smuggled axioms in those closures.
  2. RUN-TIME (`lake exe gxcheck`): loads the built environment and
       a. audits every theorem declared by every imported `Gxptx.*` module,
          including private/mangled theorem names by module provenance,
       b. keeps the explicit root/per-opcode citation checks,
       c. cross-checks translateStmt's opcode dispatch (parsed from
          Gxptx/Translation/Statement.lean) against `opcodeStatus`,
       d. validates every trusted/structural/gap key against its inventory,
       e. prints the per-opcode PROVEN/TRUSTED/STRUCTURAL/GAP/TRAP report and
          both inventories; exits 1 on any coverage-integrity failure.

  PASS means that the stated proof/trust/gap accounting is internally sound;
  it deliberately does not claim complete PTX ISA conformance.
-/
import Lean
import Gxptx.Semantics
import Gxptx.Verification
import Gxptx.Translation
import Gxptx.Validation
import Gxptx.Translation.Coverage

open Lean

namespace Gxptx

/-! ## Build-time axiom hygiene. -/

open Elab Command in
elab "#assert_axioms " decl:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo decl
  let axioms ← liftCoreM <| (collectAxioms name : CoreM (Array Name))
  let allowed : List Name := [``propext, ``Quot.sound]
  for ax in axioms do
    unless allowed.contains ax do
      throwError "#assert_axioms {name}: forbidden axiom {ax} (allowed: propext, Quot.sound)"
  logInfo m!"#assert_axioms {name}: OK {axioms.toList}"

#assert_axioms Gxptx.CTemplate.certified
#assert_axioms Gxptx.EmitKind.certified
#assert_axioms Gxptx.CommTemplate.certified
#assert_axioms Gxptx.CTemplate.all_complete
#assert_axioms Gxptx.CTemplate.all_count
-- Core-level operator-selection theorems (predicate/shift/div splits)
#assert_axioms Gxptx.emitNot_pred
#assert_axioms Gxptx.emitBin_pred_and
#assert_axioms Gxptx.emitBin_pred_or
#assert_axioms Gxptx.emitBin_pred_xor
#assert_axioms Gxptx.emitShr_signed
#assert_axioms Gxptx.emitDiv_signed
#assert_axioms Gxptx.emitDiv_unsigned
-- ld/st width-pairing table
#assert_axioms Gxptx.MemTy.cname_bytes_ok

/-! ## Runtime checker. -/

/-- Names always audited, beyond the per-opcode citations. -/
/- PTX definition: the manual supplies instruction rules, while this list is project validation metadata rather than an ISA operation. -/
/- Lean modeling: retaining explicit roots catches renamed or missing public certificates even when the broader provenance audit remains clean. -/
def rootCerts : List String :=
  ["Gxptx.CTemplate.certified", "Gxptx.EmitKind.certified",
   "Gxptx.CommTemplate.certified", "Gxptx.CTemplate.all_complete",
   "Gxptx.CTemplate.all_count",
   "Gxptx.emitNot_pred", "Gxptx.emitBin_pred_and", "Gxptx.emitBin_pred_or",
   "Gxptx.emitBin_pred_xor", "Gxptx.emitShr_signed", "Gxptx.emitShr_signed64",
   "Gxptx.emitShr_unsigned", "Gxptx.emitShr_unsigned64",
   "Gxptx.emitDiv_signed", "Gxptx.emitDiv_unsigned", "Gxptx.emitRem_signed",
   "Gxptx.emitMin_signed", "Gxptx.emitMax_signed",
   "Gxptx.MemTy.cname_bytes_ok", "Gxptx.Ptx.mulWideU_exact",
   "Gxptx.CMov.unpack_pack_lo", "Gxptx.CMov.unpack_pack_hi"]

/- PTX definition: the ISA does not prescribe proof-assistant foundations; this is the project's theorem-hygiene policy. -/
/- Lean modeling: permitting only propext and Quot.sound rejects sorryAx and user axioms without excluding Lean's standard quotient reasoning. -/
def allowedAxioms : List Name := [``propext, ``Quot.sound]

/- PTX definition: axiom collection validates proofs about ISA rules but is itself Lean metaprogramming outside the instruction semantics. -/
/- Lean modeling: Lean's kernel environment is the authoritative source for proof dependencies, including indirect and private theorem use. -/
def collectFor (env : Environment) (n : Name) : IO (Array Name) := do
  let ctx : Core.Context := { fileName := "<gxcheck>", fileMap := default }
  let st : Core.State := { env := env }
  let (axs, _) ← ((collectAxioms n : CoreM (Array Name)).toIO ctx st)
  return axs

/-- Extract the opcodes translateStmt dispatches on: the top-level match arms
    `      | "op" [| "op2"] =>` (exactly 6-space indent) of
    Translation/Statement.lean. -/
/- PTX definition: each extracted token names an instruction family whose accepted forms must be accounted against the manual. -/
/- Lean modeling: parsing the actual match arms prevents the proof/trust/gap table from silently drifting away from executable dispatch. -/
def parseDispatchOpcodes (src : String) : List String := Id.run do
  let mut out : List String := []
  let mut inMatch := false
  for line in src.splitOn "\n" do
    if (line.trimLeft).startsWith "match head with" ||
        line.trim == "let emitted := match head with" then
      inMatch := true
    else if inMatch && line.startsWith "      | \"" then
      -- take every "quoted" token before =>
      let head := (line.splitOn "=>").headD ""
      let parts := head.splitOn "\""
      -- parts: [prefix, op1, sep, op2, ...] — odd indices are the opcodes
      let mut i := 0
      for p in parts do
        if i % 2 == 1 then out := out ++ [p]
        i := i + 1
    else if inMatch && line.startsWith "def " then
      inMatch := false
  return out

#guard parseDispatchOpcodes "      let emitted := match head with\n      | \"dp2a\" | \"dp4a\" => x\n        | \"not_an_opcode\" => y\n" == ["dp2a", "dp4a"]
#guard parseDispatchOpcodes "      match head with\n      | \"ld\" | \"ldu\" => x\n" == ["ld", "ldu"]

/- PTX definition: this command checks the project's accounting of modeled ISA rules and explicitly does not assert full ISA conformance. -/
/- Lean modeling: loading all four aggregate modules makes theorem discovery explicit instead of relying on accidental translator import paths. -/
def main : IO UInt32 := do
  -- load the built environment
  initSearchPath (← findSysroot)
  searchPathRef.modify (fun sp => System.FilePath.mk ".lake/build/lib" :: sp)
  let env ← importModules #[{ module := `Gxptx.Semantics },
    { module := `Gxptx.Verification }, { module := `Gxptx.Translation },
    { module := `Gxptx.Validation }] {}
  let mut fails : Nat := 0
  let mut citationChecks : Nat := 0
  let invKeys := trustedInventory.map (·.1)
  let gapKeys := knownGapInventory.map (·.1)
  let allTheorems := allGxptxTheorems env
  -- Parsing and the compiler/toolchain apply to the whole translation rather
  -- than one opcode row. Every other trust key must be cited by a dispatched
  -- opcode (or by the explicit `trap` status) so stale assumptions cannot
  -- survive unnoticed in the machine-readable inventory.
  let globalTrustedKeys := ["parsing", "toolchain"]
  let mut seenTrustedKeys : List String := globalTrustedKeys
  let mut seenGapKeys : List String := []

  IO.println "== gxcheck: verification coverage report =="
  IO.println "PROVEN means a cited shallow Lean value-model theorem; exact emitted-C spelling/compiler behavior remains TRUSTED(template-print)."

  -- (a) Axiom hygiene on EVERY theorem owned by every imported Gxptx module.
  -- Module provenance includes declarations with `_private`/mangled names.
  if env.header.moduleNames.size != env.header.moduleData.size then
    IO.println s!"FAIL imported module-name/data tables differ in size ({env.header.moduleNames.size} vs {env.header.moduleData.size})"
    fails := fails + 1
  if allTheorems.isEmpty then
    IO.println "FAIL no theorem ConstantInfo found in imported Gxptx modules"
    fails := fails + 1
  for n in allTheorems do
    let axs ← collectFor env n
    let bad := axs.filter (fun a => ! allowedAxioms.contains a)
    if bad.size > 0 then
      IO.println s!"FAIL theorem {n}: forbidden axioms {bad.toList.map (·.toString)}"
      fails := fails + 1
  IO.println s!"imported Gxptx modules: {gxptxModuleCount env}; unique theorem ConstantInfo audited: {allTheorems.length}"

  -- (b) Keep explicit root-certificate existence and axiom checks.  These
  -- intentionally duplicate part of (a): they also guard stable public names.
  for nm in rootCerts do
    let n := nm.toName
    if ! env.contains n then
      IO.println s!"FAIL missing certificate {nm}"
      fails := fails + 1
    else
      let axs ← collectFor env n
      let bad := axs.filter (fun a => ! allowedAxioms.contains a)
      citationChecks := citationChecks + 1
      if bad.size > 0 then
        IO.println s!"FAIL {nm}: forbidden axioms {bad.toList.map (·.toString)}"
        fails := fails + 1

  -- Inventory keys must be unambiguous before status strings cite them.
  for (key, _) in trustedInventory do
    if (trustedInventory.filter (fun entry => entry.1 == key)).length != 1 then
      IO.println s!"FAIL duplicate trustedInventory key '{key}'"
      fails := fails + 1
  for (key, _) in knownGapInventory do
    if (knownGapInventory.filter (fun entry => entry.1 == key)).length != 1 then
      IO.println s!"FAIL duplicate knownGapInventory key '{key}'"
      fails := fails + 1
  for key in globalTrustedKeys do
    if !invKeys.contains key then
      IO.println s!"FAIL global trusted key '{key}' is not in trustedInventory"
      fails := fails + 1
  for (op, _) in opcodeStatus do
    if (opcodeStatus.filter (fun entry => entry.1 == op)).length != 1 then
      IO.println s!"FAIL duplicate opcodeStatus entry '{op}'"
      fails := fails + 1

  -- (c,d,e) Opcode dispatch coverage plus trust/gap-key integrity.
  let transSrc ← IO.FS.readFile "Gxptx/Translation/Statement.lean"
  let dispatched := parseDispatchOpcodes transSrc
  let tableOps := opcodeStatus.map (·.1)
  for op in dispatched do
    match opcodeStatus.find? (·.1 == op) with
    | none =>
      IO.println s!"FAIL opcode '{op}' dispatched by translateStmt but has NO opcodeStatus entry — UNVERIFIED"
      fails := fails + 1
    | some (_, status) =>
      let mut label := ""
      let mut bad := false
      for part in status.splitOn ";" do
        if part.startsWith "proven:" then
          let thms := (part.drop 7).splitOn ","
          for t in thms do
            let n := t.toName
            if ! env.contains n then
              IO.println s!"FAIL opcode '{op}': cited theorem {t} does not exist"
              bad := true
            else
              let axs ← collectFor env n
              citationChecks := citationChecks + 1
              let badAx := axs.filter (fun a => ! allowedAxioms.contains a)
              if badAx.size > 0 then
                IO.println s!"FAIL opcode '{op}': theorem {t} uses forbidden axioms {badAx.toList.map (·.toString)}"
                bad := true
          label := label ++ s!"PROVEN({(part.drop 7)}) "
        else if part.startsWith "trusted:" || part.startsWith "structural:" then
          let key := (part.splitOn ":").getLastD ""
          if ! invKeys.contains key then
            IO.println s!"FAIL opcode '{op}': trusted key '{key}' not in trustedInventory"
            bad := true
          else if !seenTrustedKeys.contains key then
            seenTrustedKeys := key :: seenTrustedKeys
          let tag := if part.startsWith "trusted:" then "TRUSTED" else "STRUCTURAL"
          label := label ++ s!"{tag}({key}) "
        else if part.startsWith "gap:" then
          let key := part.drop 4
          if key.isEmpty then
            IO.println s!"FAIL opcode '{op}': empty gap key"
            bad := true
          else if ! gapKeys.contains key then
            IO.println s!"FAIL opcode '{op}': gap key '{key}' not in knownGapInventory"
            bad := true
          else
            if !seenGapKeys.contains key then seenGapKeys := key :: seenGapKeys
          label := label ++ s!"GAP({key}) "
        else if part == "trap" then
          if !invKeys.contains "trap" then
            IO.println s!"FAIL opcode '{op}': trap status has no trustedInventory entry"
            bad := true
          else if !seenTrustedKeys.contains "trap" then
            seenTrustedKeys := "trap" :: seenTrustedKeys
          label := label ++ "TRAP "
        else
          IO.println s!"FAIL opcode '{op}': unparseable status '{part}'"
          bad := true
      if bad then fails := fails + 1
      IO.println s!"  {op}: {label.trimRight}"
  for op in tableOps do
    if ! dispatched.contains op then
      IO.println s!"FAIL opcodeStatus lists '{op}' but translateStmt does not dispatch it (stale entry)"
      fails := fails + 1

  -- Except for the two explicit global boundaries above, every trust entry
  -- must remain attached to an opcode/status. This is the trust-inventory
  -- analogue of the known-gap stale-entry check below.
  for key in invKeys do
    if !seenTrustedKeys.contains key then
      IO.println s!"FAIL trusted boundary '{key}' is not cited by any dispatched opcode and is not global"
      fails := fails + 1

  -- Every gap discovered in the PDF audit must remain attached to at least
  -- one dispatched opcode; otherwise its inventory entry has gone stale.
  for key in gapKeys do
    if !seenGapKeys.contains key then
      IO.println s!"FAIL known gap '{key}' is not cited by any dispatched opcode"
      fails := fails + 1

  -- `all_complete` proves at elaboration time that no constructor is absent;
  -- `all_count` fixes the expected 71 concrete values.  Runtime duplicate
  -- rejection closes the remaining list-representation invariant.
  let cTemplateCount := CTemplate.all.length
  if cTemplateCount != 71 then
    IO.println s!"FAIL CTemplate.all has {cTemplateCount} entries; theorem-backed expected count is 71"
    fails := fails + 1
  for t in CTemplate.all do
    if (CTemplate.all.filter (fun candidate => candidate == t)).length != 1 then
      IO.println s!"FAIL duplicate CTemplate.all entry {repr t}"
      fails := fails + 1
  IO.println s!"concrete CTemplate inventory: {cTemplateCount} (complete theorem + duplicate check)"

  -- EmitKind enumeration completeness (report side)
  for k in EmitKind.all do
    let n := (EmitKind.thmName k).toName
    if ! env.contains n then
      IO.println s!"FAIL EmitKind {repr k}: theorem {EmitKind.thmName k} missing"
      fails := fails + 1

  -- Communication-template enumeration completeness (report side).
  for k in CommTemplate.all do
    let n := (CommTemplate.thmName k).toName
    if ! env.contains n then
      IO.println s!"FAIL CommTemplate {repr k}: theorem {CommTemplate.thmName k} missing"
      fails := fails + 1

  IO.println ""
  IO.println "== TRUSTED-INVENTORY =="
  for (k, j) in trustedInventory do
    IO.println s!"  {k}: {j}"

  IO.println ""
  IO.println "== KNOWN-GAP-INVENTORY (PTX ISA 9.3) =="
  for (k, j) in knownGapInventory do
    IO.println s!"  {k}: {j}"

  IO.println ""
  IO.println s!"opcodes dispatched: {dispatched.length}; unique Gxptx theorems audited: {allTheorems.length}; explicit citation checks: {citationChecks}; known gaps: {knownGapInventory.length}; failures: {fails}"
  if fails == 0 then
    IO.println "gxcheck: PASS (coverage metadata is consistent; all imported Gxptx theorem closures use only {propext, Quot.sound}; known gaps are reported — this is NOT a claim of full PTX ISA conformance)"
    return 0
  else
    IO.println "gxcheck: FAIL"
    return 1

end Gxptx

/- PTX definition: the executable reports verification coverage for the modeled subset and never claims complete manual conformance. -/
/- Lean modeling: direct delegation preserves the audit's UInt32 exit status so CI cannot turn a recorded failure into success. -/
def main : IO UInt32 := Gxptx.main
