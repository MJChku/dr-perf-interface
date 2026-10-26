/-
  Gxptx.Translation.Statement — one-statement PTX-to-C dispatch.
  Opcode arms remain in their historical family order; scalar arithmetic
  variants are routed explicitly and unsupported forms fail closed.
-/
import Gxptx.Translation.Parsing
import Gxptx.Translation.C.Calls
import Gxptx.Translation.C.Expression
import Gxptx.Translation.C.Operands
import Gxptx.Translation.C.Templates
import Gxptx.Semantics.C.Memory
import Gxptx.Translation.Latency
set_option linter.deprecated false
namespace Gxptx

/- PTX definition: vector-source mov packs and vector-destination mov unpacks; brace location, not operand names, determines direction. -/
/- Lean modeling: syntax metadata is retained before normalization and threaded through label/predicate recursion; parser/printer correctness remains trusted. -/
private def movVectorPack? (stmt : String) : Option Bool :=
  match (stmt.replace "\t" " " |>.replace "\n" " ").splitOn "mov." with
  | [_before, after] =>
    let ty := (after.splitOn " ").headD ""
    let args := strim (after.drop ty.length)
    match splitArgs args with
    | [d, a] =>
      if d.startsWith "{" && d.endsWith "}" && !a.contains '{' then some false
      else if a.startsWith "{" && a.endsWith "}" && !d.contains '{' then some true
      else none
    | _ => none
  | _ => none

/- PTX definition: ldmatrix has one, two or four destination registers followed by a bracketed shared-memory address. -/
/- Lean modeling: preserve ldmatrix brace syntax before normalization, including through label and predicate recursion; scalar operand lists do not qualify. -/
private def matrixVectorSyntax (stmt : String) (load : Bool) : Bool :=
  match (stmt.replace "\t" " " |>.replace "\n" " ").splitOn (if load then "ldmatrix." else "stmatrix.") with
  | [_before, after] =>
    let modifiers := (after.splitOn " ").headD ""
    let raw := strim (after.drop modifiers.length)
    let args := if raw.endsWith ";" then strim (raw.dropRight 1) else raw
    match splitArgs args with
    | [first, second] =>
        let d := if load then first else second
        let a := if load then second else first
        d.startsWith "{" && d.endsWith "}" &&
        a.startsWith "[" && a.endsWith "]" &&
        [1, 2, 4].contains (splitArgs (d.replace "{" "" |>.replace "}" "")).length
    | _ => false
  | _ => false


/- PTX definition: `wgmma.mma_async` takes an accumulator brace vector `d`, then either an A descriptor or a four-register A brace vector, a B descriptor and scalar scale/transpose operands. -/
/- Lean modeling: record (accumulator count, A-vector count or 0 for a descriptor) before brace normalization, and through label/predicate recursion, so a flattened malformed operand list cannot be re-split into an accepted form. -/
private def wgmmaVectorSyntax (stmt : String) : Option (Nat × Nat) :=
  match (stmt.replace "\t" " " |>.replace "\n" " ").splitOn "wgmma.mma_async." with
  | [_before, after] =>
    let modifiers := (after.splitOn " ").headD ""
    let raw := strim (after.drop modifiers.length)
    let args := if raw.endsWith ";" then strim (raw.dropRight 1) else raw
    let vector? (v : String) : Option Nat :=
      let inside := strim (v.drop 1 |>.dropRight 1)
      if v.startsWith "{" && v.endsWith "}" && !inside.contains '{' && !inside.contains '}' then
        some (splitArgs inside).length
      else none
    match splitArgs args with
    | d :: a :: b :: rest =>
      if b.contains '{' || rest.any (·.contains '{') then none
      else match vector? d with
        | some nd =>
          if a.contains '{' then (vector? a).map (fun na => (nd, na)) else some (nd, 0)
        | none => none
    | _ => none
  | _ => none

/- PTX definition: WMMA f16 operands have eight registers; bf16 A/B arities depend on the tile shape, with eight f32 accumulators. -/
/- Lean modeling: retain vector arity before normalization and through recursion; malformed flattened operand lists fail closed. -/
private def wmmaVectorSyntax (stmt : String) : Bool :=
  match (stmt.replace "\t" " " |>.replace "\n" " ").splitOn "wmma." with
  | [_before, after] =>
    let mods := (after.splitOn " ").headD ""
    let raw := strim (after.drop mods.length)
    let raw := if raw.endsWith ";" then strim (raw.dropRight 1) else raw
    let bf := (mods.splitOn ".").contains "bf16"
    let na := if !bf then 8 else if (mods.splitOn ".").contains "m8n32k16" then 2 else if (mods.splitOn ".").contains "m32n8k16" then 8 else 4
    let nb := if !bf then 8 else if (mods.splitOn ".").contains "m8n32k16" then 8 else if (mods.splitOn ".").contains "m32n8k16" then 2 else 4
    let vec (n : Nat) (v : String) := v.startsWith "{" && v.endsWith "}" &&
      (splitArgs (v.drop 1 |>.dropRight 1)).length == n
    match splitArgs raw with
    | [d, a] | [d, a, _] =>
        if mods.startsWith "load." then
          vec (if mods.startsWith "load.a." then na else if mods.startsWith "load.b." then nb else 8) d && a.startsWith "["
        else if mods.startsWith "store." then d.startsWith "[" && vec 8 a else false
    | [d, a, b, c] => mods.startsWith "mma." && vec 8 d && vec na a && vec nb b && vec 8 c
    | _ => false
  | _ => false

/- PTX definition: each instruction opcode and modifier sequence determines operand shape and an operation rule. -/
/- Lean modeling: dispatch selects value-model-backed templates or explicit traps; theorems are shallow value equalities while printing, compilation, and inventoried limitations remain separate boundaries. -/
partial def translateStmtScoped (stmt : String) (guardGlobal : Bool)
    (regs : Std.HashSet String := {}) (movVector : Option Bool := none)
    (ldmatrixVector : Bool := false) (wgmmaShape : Option (Nat × Nat) := none) (wmmaVector : Bool := false) (profile : Bool := true)
    (latencyPredicates : List String := []) : String :=
  let movVector := movVector.orElse fun _ => movVectorPack? stmt
  let ldmatrixVector := ldmatrixVector || matrixVectorSyntax stmt true || matrixVectorSyntax stmt false
  let wgmmaShape := wgmmaShape.orElse fun _ => wgmmaVectorSyntax stmt
  let wmmaVector := wmmaVector || wmmaVectorSyntax stmt
  -- normalise whitespace (PTX separates with tabs/newlines); strip only the
  -- LEADING/TRAILING inline-asm block braces `{ … }` — internal `{…}` vector
  -- operands must survive
  -- normalise whitespace; strip inline-asm block braces. (Vector `{a,b,c,d}`
  -- operands are recovered by splitArgs/operand from the comma-separated
  -- remainder — ld/st reconstruct the element offsets.)
  let s0 := strim ((((stmt.replace "\t" " ").replace "\n" " ").replace "{" " ").replace "}" " ")
  -- ptxas accepts an opcode directly followed by its memory operand
  -- (`tcgen05.st...b32[%r1], ...`); separate them so the opcode is one token.
  -- Predicated/labelled statements reach this split through recursion.
  let s := match s0.splitOn "[" with
    | op :: rest@(_ :: _) =>
      if !op.isEmpty && !op.contains ' ' && !(op.replace "::" "").contains ':' && !op.startsWith "@" then
        op ++ " [" ++ "[".intercalate rest
      else s0
    | _ => s0
  -- A colon is a delimiter, including `name :` and `name:instruction`.
  -- Address-space qualifiers such as shared::cta are not label prefixes.
  let labelPrefix := (s.splitOn ":").headD ""
  let label := strim labelPrefix
  let afterColon := strim (s.drop (labelPrefix.length + 1))
  let hasLabel := labelPrefix.length < s.length && !label.isEmpty &&
    label.toList.all (fun c => c.isAlphanum || c == '_' || c == '$' || c == '.' || c == '%') &&
    !afterColon.startsWith ":"
  if s.isEmpty || s.startsWith "//" || s.startsWith "." then ""
  else if hasLabel then
    -- Named call-prototype/target declarations carry static call metadata;
    -- unlike branch labels, they have no executable position in the body.
    if afterColon.startsWith ".callprototype" || afterColon.startsWith ".calltargets" then ""
    else
      let lbl := label.replace "$" ""
      let rest := translateStmtScoped afterColon guardGlobal regs movVector ldmatrixVector wgmmaShape wmmaVector profile latencyPredicates
      if rest.isEmpty then s!"  L_{lbl}: ;" else s!"  L_{lbl}: ;\n{rest}"
  else if s.startsWith "@" then
    -- predicated: `@[!]%p <instr>` → `if ([!]R_p) { <instr> }`
    let r0 := s.drop 1
    let neg := r0.startsWith "!"
    let r1 := if neg then r0.drop 1 else r0
    match (r1.splitOn " ").filter (! ·.isEmpty) with
    | pred :: _ =>
      let inner := strim (r1.drop pred.length)
      let g := (operand pred).toC
      let gg := if neg then s!"(!{g})" else g
      let deps := latencyPredicates ++ latencyRegisters regs pred
      let line := strim (translateStmtScoped inner guardGlobal regs movVector ldmatrixVector wgmmaShape wmmaVector profile deps)
      -- A disabled instruction issues and consumes its predicate, but neither
      -- waits on data operands nor writes any destination shadow/value.
      -- Keep disabled issue attribution at the same static site as the
      -- instruction counter in the enabled arm, despite emitted newlines.
      let offOp := (inner.splitOn " ").headD ""
      let offLineDelta := (line.splitOn "\n").length - 1
      let offDeps := String.intercalate ", " (deps.eraseDups.map latencyShadow)
      let disabled := if profile && !line.startsWith "drperf_gpu_device_" then
        " else { gx_latency_off(__func__, " ++ s!"\"{offOp}\", __LINE__ - {offLineDelta}, " ++
        "{" ++ offDeps ++ "}); }" else ""
      if line.isEmpty then "" else "    if (" ++ gg ++ ") { " ++ line ++ " }" ++ disabled
    | _ => ""
  else
    let parts := s.splitOn " " |>.filter (! ·.isEmpty)
    match parts with
    | [] => ""
    | op :: _ =>
      let rest := strim (s.drop op.length)
      let args := splitArgs rest
      let toks := op.splitOn "."
      let head := toks.headD ""
      -- the type tag carried by the instruction, if any
      let ty : Ty := (toks.filterMap parseTy?).headD .b32
      -- parseTy? has no 128-bit Ty; keep b128 visible to mov/accounting so
      -- 16-byte moves and stores are neither mis-typed nor charged as 4 bytes.
      let tyTok := if toks.contains "b128" then "b128"
                   else (toks.filter (parseTy? · |>.isSome)).getLastD "b32"
      let isShared := toks.any (fun t => t == "shared" || t.startsWith "shared::")
      -- generic/global space may hit fake device buffers → guard; shared,
      -- local, param, const are real host memory → direct access.
      let deviceAddress := ! (isShared || toks.contains "local"
                        || toks.any (fun t => t == "param" || t.startsWith "param::")
                        || toks.contains "const")
      let guarded := guardGlobal && deviceAddress
      let operand := registerOperand regs
      let oC (x : String) := (operand x).toC
      let fC := floatOperand regs tyTok
      let fWrite := writeFloat tyTok
      let vol := if toks.any (fun q => q == "volatile" || q == "relaxed"
                              || q == "acquire" || q == "release") then "volatile " else ""
      -- Emission kind: modeled scalar integers use width-indexed templates backed
      -- by shallow value equalities. Predicate routes retain their reviewed shapes;
      -- floating arithmetic emits only through each opcode's narrow allowlist and
      -- every other arithmetic float spelling traps.
      let k : EK := if ty.isInt then .int (if ty.is64 then .w64 else .w32) else .f
      -- As on hardware, a shared address carries the owning CTA's cluster
      -- rank in bits [24,32) (cvta.to.shared, mapa). shared::cta accesses use
      -- the low 24 offset bits; shared::cluster accesses decode the rank.
      let clusterShared := toks.contains "shared::cluster"
      let sharedPtr (a : String) : String :=
        if clusterShared then s!"((uint64_t)gx_cluster_shared_ptr((uint32_t)({a})))"
        else s!"((uint64_t)(uintptr_t)(_gx_shm + (({a}) & 0xFFFFFFu)))"
      let mkAddr (mem : String) : String :=
        let a := parseAddr mem regs
        if isShared then sharedPtr a else a
      let loadE (mct : String) (addr : String) : String :=
        if guarded then s!"gx_gld<{mct}>({addr})"
        else if deviceAddress then
          s!"(gx_fake({addr}) ? ({mct})0 : *({mct} const {vol}*)(uintptr_t)({addr}))"
        else s!"*({mct} const {vol}*)(uintptr_t)({addr})"
      let trapOp := s!"    gx_ptx_trap_op(\"{op}\");"
      let bin (mk : CExpr → CExpr → CExpr) : String :=
        match args with
        | [d, a, b] =>
          if tyTok == "f32" || tyTok == "f64" then
            if d == "_" then trapOp
            else fWrite d ((mk (.var (fC a)) (.var (fC b))).toC .f)
          else s!"    {dest d} = {(mk (operand a) (operand b)).toC k};"
        | _ => trapOp
      let binPrint (print : String → String → String) : String :=
        match args with
        | [d, a, b] =>
          if tyTok == "f32" || tyTok == "f64" then
            if d == "_" then trapOp else fWrite d (print (fC a) (fC b))
          else s!"    {dest d} = {print (oC a) (oC b)};"
        | _ => trapOp
      -- General floating add/sub/mul/fma: rounding (rn/rz/rm/rp), .ftz and
      -- .sat for f32/f64, packed .f32x2 lanes and mixed-precision
      -- .f32.{f16,bf16} sources, through the gx_farith32/64 runtime helpers.
      let floatArith : Option String :=
        let opCode := match head with
          | "add" => some 0 | "sub" => some 1 | "mul" => some 2 | "fma" => some 3 | _ => none
        let mods := toks.drop 1
        let typesOf := mods.filter (fun t => ["f32", "f64", "f32x2", "f16", "bf16"].contains t)
        let quals := mods.filter (fun t => !typesOf.contains t)
        let rnd := quals.filter (fun t => ["rn", "rz", "rm", "rp"].contains t)
        let okQuals := quals.all (fun t => ["rn", "rz", "rm", "rp", "ftz", "sat"].contains t) &&
          rnd.length ≤ 1 && (quals.filter (· == "ftz")).length ≤ 1 && (quals.filter (· == "sat")).length ≤ 1
        let rcode := match rnd.headD "rn" with | "rz" => 1 | "rm" => 2 | "rp" => 3 | _ => 0
        let flags := rcode + (if quals.contains "ftz" then 4 else 0) + (if quals.contains "sat" then 8 else 0)
        let f32 (x : String) := s!"(uint32_t)({floatBitsOperand regs "f32" x})"
        let half (ty x : String) :=
          s!"gx_f_as<uint32_t>({if ty == "bf16" then "gx_bf2f" else "gx_h2f"}((uint16_t)({oC x})))"
        match opCode with
        | none => none
        | some code =>
          if !okQuals || (head == "fma" && rnd.isEmpty) then none
          else match typesOf, args with
            | ["f32"], [d, a, b] =>
              if code == 3 || d == "_" then none
              else some s!"    gx_movb({dest d}, (uint32_t)gx_farith32({code}u, {f32 a}, {f32 b}, 0u, {flags}u));"
            | ["f32"], [d, a, b, c] =>
              if code != 3 || d == "_" then none
              else some s!"    gx_movb({dest d}, (uint32_t)gx_farith32(3u, {f32 a}, {f32 b}, {f32 c}, {flags}u));"
            | ["f64"], [d, a, b] =>
              if code == 3 || d == "_" || quals.contains "ftz" || quals.contains "sat" then none
              else some s!"    gx_movb({dest d}, (uint64_t)gx_farith64({code}u, (uint64_t)({floatBitsOperand regs "f64" a}), (uint64_t)({floatBitsOperand regs "f64" b}), 0ull, {flags}u));"
            | ["f64"], [d, a, b, c] =>
              if code != 3 || d == "_" || quals.contains "ftz" || quals.contains "sat" then none
              else some s!"    gx_movb({dest d}, (uint64_t)gx_farith64(3u, (uint64_t)({floatBitsOperand regs "f64" a}), (uint64_t)({floatBitsOperand regs "f64" b}), (uint64_t)({floatBitsOperand regs "f64" c}), {flags}u));"
            | ["f32x2"], d :: rest =>
              -- Lane 0 is the low 32 bits of each 64-bit operand.
              let lane (x : String) (i : Nat) := s!"(uint32_t)(((uint64_t)({oC x})) >> {32 * i})"
              let one (i : Nat) := match rest with
                | [a, b] => s!"(uint64_t)gx_farith32({code}u, {lane a i}, {lane b i}, 0u, {flags}u)"
                | [a, b, c] => s!"(uint64_t)gx_farith32(3u, {lane a i}, {lane b i}, {lane c i}, {flags}u)"
                | _ => ""
              if d == "_" || quals.contains "sat" || (code == 3) != (rest.length == 3) || rest.length < 2 || rest.length > 3 then none
              else some s!"    gx_movb({dest d}, (uint64_t)({one 0} | ({one 1} << 32)));"
            | ["f32", src], [d, a, c] =>
              if code > 1 || d == "_" || quals.contains "ftz" || !(src == "f16" || src == "bf16") then none
              else some s!"    gx_movb({dest d}, (uint32_t)gx_farith32({code}u, {half src a}, {f32 c}, 0u, {flags}u));"
            | ["f32", src], [d, a, b, c] =>
              if code != 3 || d == "_" || quals.contains "ftz" || !(src == "f16" || src == "bf16") then none
              else some s!"    gx_movb({dest d}, (uint32_t)gx_farith32(3u, {half src a}, {half src b}, {f32 c}, {flags}u));"
            | _, _ => none
      -- Packed integer SIMD (PTX §9.7.1): add{.sat} on .{u,s}16x2 / .{u,s}8x4,
      -- sub{.sat} on .{u,s}8x4, min/max{.relu} on all four, via gx_simd_int.
      let simdTy := toks.filter (fun t => ["u16x2", "s16x2", "u8x4", "s8x4"].contains t)
      let simdRoute : Option String :=
        let code := match head with | "add" => some 0 | "sub" => some 1 | "min" => some 2 | "max" => some 3 | _ => none
        match code, simdTy, args with
        | some c, [ty], [d, a, b] =>
          let quals := (toks.drop 1).filter (· != ty)
          let sat := quals.contains "sat"
          let relu := quals.contains "relu"
          let ok := quals.all (fun q => q == "sat" || q == "relu") && quals.length ≤ 1 &&
            (!sat || c ≤ 1) && (!relu || (c ≥ 2 && ty.startsWith "s")) && d != "_" &&
            -- sub has no .{u,s}16x2 form (PTX §9.7.1.2 .type2 = { .s32, .u8x4, .s8x4 }).
            !(c == 1 && ty.endsWith "16x2")
          if !ok then none
          else
            let lanes := if ty.endsWith "x2" then 2 else 4
            some s!"    {dest d} = gx_simd_int({c}u, (uint32_t)({oC a}), (uint32_t)({oC b}), {lanes}u, {if ty.startsWith "s" then "true" else "false"}, {if sat then "true" else "false"}, {if relu then "true" else "false"});"
        | _, _, _ => none
      -- Generic atom/red (PTX §9.7.14.5-6, Table 35) through gx_atom_rmw.
      let atomRoute (withDest : Bool) : Option String :=
        let ops := ["and", "or", "xor", "add", "inc", "dec", "min", "max", "exch", "cas"]
        let types := ["b16", "b32", "b64", "b128", "u32", "u64", "s32", "s64", "f32", "f64", "f16", "f16x2", "bf16", "bf16x2"]
        let quals := ["relaxed", "acquire", "release", "acq_rel", "cta", "cluster", "gpu", "sys",
                      "global", "shared", "shared::cta", "shared::cluster", "noftz", "L2::cache_hint"]
        let opQ := (toks.drop 1).filter (fun t => ops.contains t)
        let tyQ := (toks.drop 1).filter (fun t => types.contains t)
        let other := (toks.drop 1).filter (fun t => !ops.contains t && !types.contains t)
        let hint := toks.contains "L2::cache_hint"
        let count (xs : List String) := (other.filter (fun t => xs.contains t)).length
        let classesOK := count ["relaxed", "acquire", "release", "acq_rel"] ≤ 1 &&
          count ["cta", "cluster", "gpu", "sys"] ≤ 1 &&
          count ["global", "shared", "shared::cta", "shared::cluster"] ≤ 1 &&
          other.eraseDups.length == other.length &&
          -- red has no destination and only .relaxed/.release semantics.
          (withDest || !other.any (fun t => t == "acquire" || t == "acq_rel"))
        match opQ, tyQ with
        | [aop], [ty] =>
          if !other.all (fun t => quals.contains t) || !classesOK || toks.getLastD "" != ty then none
          else
            let halfTy := ["f16", "f16x2", "bf16", "bf16x2"].contains ty
            let code : Option Nat := match aop, ty with
              | "and", "b32" | "and", "b64" => some 0
              | "or", "b32" | "or", "b64" => some 1
              | "xor", "b32" | "xor", "b64" => some 2
              | "add", "u32" | "add", "s32" | "add", "u64" => some 3
              | "inc", "u32" => some 4
              | "dec", "u32" => some 5
              | "min", "u32" | "min", "s32" | "min", "u64" | "min", "s64" => some 6
              | "max", "u32" | "max", "s32" | "max", "u64" | "max", "s64" => some 7
              | "add", "f32" => some (if toks.contains "global" then 9 else 8)
              | "add", "f64" => some 10
              | "add", "f16" => some 11 | "add", "f16x2" => some 12
              | "add", "bf16" => some 13 | "add", "bf16x2" => some 14
              | "exch", "b32" | "exch", "b64" => some 15
              | _, _ => none
            let ct := match ty with
              | "b16" | "f16" | "bf16" => "uint16_t" | "b32" | "u32" | "f32" | "f16x2" | "bf16x2" => "uint32_t"
              | "s32" => "int32_t" | "s64" => "int64_t" | _ => "uint64_t"
            let val (x : String) := if ty == "f32" || ty == "f64" then s!"({ct})({floatBitsOperand regs ty x})" else s!"({ct})({oC x})"
            let rest := if hint then args.dropLast else args
            let hdr := if withDest then 1 else 0
            if halfTy && !toks.contains "noftz" then none
            else if aop == "cas" then
              match rest with
              | [d, m, b, c] =>
                if !withDest || d == "_" || !["b16", "b32", "b64", "b128"].contains ty then none
                else if ty == "b128" then some s!"    {dest d} = gx_atom_b128({mkAddr m}, true, (unsigned __int128)({oC b}), (unsigned __int128)({oC c}));"
                else some s!"    {dest d} = gx_atom_cas<{ct}>({mkAddr m}, ({ct})({oC b}), ({ct})({oC c}));"
              | _ => none
            else if aop == "exch" && ty == "b128" then
              match rest with
              | [d, m, b] => if !withDest || d == "_" then none
                             else some s!"    {dest d} = gx_atom_b128({mkAddr m}, false, (unsigned __int128)({oC b}), 0);"
              | _ => none
            else match code, rest with
              | some c, [d, m, b] =>
                if !withDest || d == "_" then none
                else some s!"    gx_movb({dest d}, gx_atom_rmw<{ct}>({mkAddr m}, {c}u, {val b}));"
              | some c, [m, b] =>
                if withDest || hdr != 0 then none
                else some s!"    (void)gx_atom_rmw<{ct}>({mkAddr m}, {c}u, {val b});"
              | _, _ => none
        | _, _ => none
      let scalarInt := (scalarArithTy? tyTok).isSome
      let plainScalarInt := scalarInt && toks.length == 2
      let emitted := match head with
      -- ==================================================================
      -- CONTROL-FLOW PRELUDE AND DATA MOVEMENT
      -- ==================================================================
      /- PTX definition: `ret` returns control from the current function to its caller. -/
      /- Lean modeling: a C `return;` preserves the supported void-function control transfer; function ABI and compiler behavior are structural trusted boundaries. -/
      | "ret"  => "    return;"
      /- PTX definition: `bra` transfers control to its target label, optionally under an instruction predicate. -/
      /- Lean modeling: scope alpha-renaming and C `goto` preserve direct branch topology. -/
      | "bra"  => match args with
                  | l :: _ =>
                    let lbl := l.replace "$" ""
                    s!"    goto L_{lbl};"
                  | _ => ""
      /- PTX definition: `mov` copies a scalar bit pattern or packs/unpacks vector components according to the declared width. -/
      /- Lean modeling: scalar/two-half helpers follow shallow `CMov` equalities; printing is trusted, while `gap:mov-forms`, `gap:special-register-coverage`, and sink handling delimit the route. -/
      | "mov"  =>
        -- The vector table is b16=2x8, b32=2x16/4x8,
        -- b64=2x32/4x16. b128 remains outside this scalar ABI.
        -- Original braces select direction; explicit instruction width/count
        -- selects the helper. Actual C local sizes validate vector fields.
        let movTypes := ["pred", "b16", "b32", "b64", "b128", "u16", "u32", "u64",
                         "s16", "s32", "s64", "f32", "f64"]
        -- b128 (PTX 8.3+): copy, or pack/unpack two b64 fields only.
        if toks.length != 2 || !movTypes.contains (toks.getLastD "") then trapOp
        else if let some pack := movVector then
          let count := args.length - 1
          let valid := (tyTok == "b16" && count == 2) ||
            ((tyTok == "b32" || tyTok == "b64") && (count == 2 || count == 4)) ||
            (tyTok == "b128" && count == 2)
          if !valid then trapOp
          else if pack then
            let d := args.headD "_"
            let fields := args.drop 1
            -- An immediate field has no C register width: give an integer
            -- literal the lane's unsigned type. A float literal (0f = 32-bit,
            -- 0d = 64-bit) must already match the lane width.
            let laneBits := (if tyTok == "b16" then 16 else if tyTok == "b32" then 32
                             else if tyTok == "b128" then 128 else 64) / count
            let immediate (x : String) : Bool :=
              match x.toList.head? with
              | some c => c.isDigit || c == '-'
              | none => false
            let field (x : String) : Option String :=
              if !immediate x then some (oC x)
              else if x.startsWith "0f" || x.startsWith "0F" then
                if laneBits == 32 then some (oC x) else none
              else if x.startsWith "0d" || x.startsWith "0D" then
                if laneBits == 64 then some (oC x) else none
              else some s!"(uint{laneBits}_t)({oC x})"
            let typed := fields.filterMap field
            if d == "_" || fields.contains "_" || typed.length != fields.length then trapOp
            else s!"    gx_movpack{count}<{if tyTok == "b128" then "unsigned __int128" else ctypeOf tyTok}>({dest d}, {String.intercalate ", " typed});"
          else
            let fields := args.take count
            let src := args.getLastD "_"
            if src == "_" || fields.all (· == "_") then trapOp
            else
              let assigns := String.intercalate " " (fields.enum.filterMap (fun (i, d) =>
                if d == "_" then none
                else some s!"gx_movunpack_lane<{if tyTok == "b128" then "unsigned __int128" else ctypeOf tyTok}, {count}, {i}>({dest d}, _gx_mov_src);"))
              s!"    \{ auto _gx_mov_src = gx_movread<{if tyTok == "b128" then "unsigned __int128" else ctypeOf tyTok}>({oC src}); {assigns} }"
        else match args with
        | [d, a] =>
          if d == "_" then trapOp
          else if tyTok == "b128" then s!"    memcpy(&{dest d}, &{oC a}, 16);"
          else if tyTok == "b32" || tyTok == "b64" then
            s!"    {printMovB (dest d) (bitOperand regs tyTok a)}"
          else if tyTok == "f32" || tyTok == "f64" then
            fWrite d (fC a)
          else if isFloatOp d != isFloatOp a then s!"    {printMovB (dest d) (oC a)}"
          else s!"    {dest d} = {(operand a).toC};"
        | _ => trapOp
      -- ==================================================================
      -- ARITHMETIC AND MULTIPLY
      -- ==================================================================
      /- PTX definition: `add` stores the fixed-width sum of its two operands, subject to opcode modifiers; `.f16x2` and `.bf16x2` add the two packed lanes independently. -/
      /- Lean modeling: six scalar integers are proved; plain/RN f32/f64 and exact plain f16x2/bf16x2 and scalar bf16 add retain narrow trusted C routes, while other modes and formats trap. -/
      | "add"  =>
        let rawFloat := toks == ["add", "f32"] || toks == ["add", "f64"] ||
          toks == ["add", "rn", "f32"] || toks == ["add", "rn", "f64"]
        if !simdTy.isEmpty then simdRoute.getD trapOp
        else if toks.contains "f32x2" || (toks.contains "f32" && (toks.contains "f16" || toks.contains "bf16")) then floatArith.getD trapOp
        else if toks == ["add", "f16x2"] then
          binPrint (fun a b => CTemplate.addF16x2.print a b "")
        else if toks == ["add", "bf16x2"] then
          binPrint (fun a b => s!"gx_bf16x2_add({a}, {b})")
        else if toks == ["add", "bf16"] then
          binPrint (fun a b => s!"(uint16_t)gx_bf16x2_add((uint16_t)({a}), (uint16_t)({b}))")
        else if plainScalarInt || rawFloat then bin .add else floatArith.getD trapOp
      /- PTX definition: `abs` returns the magnitude of a signed integer or floating operand for supported types. -/
      /- Lean modeling: signed integers use modular `printAbsS`; floating forms use trusted sign-bit helpers (gx_absneg_*), not a floating-value theorem. -/
      | "abs"  =>
        match scalarArithTy? tyTok with
        | some (true, w) =>
          if toks.length == 2 then
            match args with
            | [d, a] => s!"    {dest d} = {printAbsS w (oC a)};"
            | _ => trapOp
          else trapOp
        | _ =>
          -- Floating abs{.ftz}.f32, abs.f64, abs{.ftz}.{f16,f16x2}, abs.{bf16,bf16x2}:
          -- sign-bit operations (f64 NaN passes through unchanged), .ftz flushes
          -- subnormals to sign-preserving zero. Any other spelling traps.
          match args with
          | [d, a] =>
            let mods := toks.drop 1
            let ftz := if mods.contains "ftz" then "1" else "0"
            if d == "_" then trapOp
            else if mods == ["f32"] || mods == ["ftz", "f32"] then
              fWrite d s!"gx_absneg_f32(0, {ftz}, {fC a})"
            else if mods == ["f64"] then fWrite d s!"gx_absneg_f64(0, {fC a})"
            else if [["f16"], ["ftz", "f16"], ["f16x2"], ["ftz", "f16x2"], ["bf16"], ["bf16x2"]].contains mods then
              let hty := mods.getLastD ""
              let lanes := if hty.endsWith "x2" then "2" else "1"
              let cast := if lanes == "2" then "uint32_t" else "uint16_t"
              let bf := if hty.startsWith "bf" then "1" else "0"
              s!"    {dest d} = ({cast})gx_absneg_h16(0, {ftz}, {bf}, {lanes}, (uint32_t)({oC a}));"
            else trapOp
          | _ => trapOp
      /- PTX definition: `sub` stores the fixed-width difference of its two operands, subject to opcode modifiers. -/
      /- Lean modeling: six scalar integers are proved; only plain/RN f32/f64 retain the trusted C route, while other modes and formats trap. -/
      | "sub"  =>
        let rawFloat := toks == ["sub", "f32"] || toks == ["sub", "f64"] ||
          toks == ["sub", "rn", "f32"] || toks == ["sub", "rn", "f64"]
        if !simdTy.isEmpty then simdRoute.getD trapOp
        else if toks.contains "f32x2" || (toks.contains "f32" && (toks.contains "f16" || toks.contains "bf16")) then floatArith.getD trapOp
        -- Binary16 subtraction is addition with each second-operand sign
        -- flipped. Reuse exact integer RN-even addition (including signed zero,
        -- subnormals, infinities and canonical NaNs), not host half arithmetic.
        else if toks == ["sub", "f16x2"] || toks == ["sub", "rn", "f16x2"] then
          binPrint (fun a b => CTemplate.addF16x2.print a s!"((uint32_t)({b}) ^ 0x80008000u)" "")
        else if plainScalarInt || rawFloat then bin .sub else floatArith.getD trapOp
      /- PTX definition: `mul` selects `.lo`, `.hi`, or `.wide` product bits according to type width and signedness. -/
      /- Lean modeling: integer modes have distinct width printers; exact plain bf16x2 and plain/RN f32/f64 retain narrow C routes while every other packed or modified form traps. -/
      | "mul"  =>
        -- Every legal integer width/mode is selected explicitly. Packed or
        -- otherwise unsupported spellings trap instead of falling through to lo.
        let rawFloat := toks == ["mul", "f32"] || toks == ["mul", "f64"] ||
          toks == ["mul", "rn", "f32"] || toks == ["mul", "rn", "f64"]
        if toks.contains "f32x2" || (toks.contains "f32" && (toks.contains "f16" || toks.contains "bf16")) then floatArith.getD trapOp
        else if toks == ["mul", "f16x2"] || toks == ["mul", "rn", "f16x2"] then
          binPrint (fun a b => s!"gx_f16x2_mul({a}, {b})")
        else if toks == ["mul", "f16"] || toks == ["mul", "rn", "f16"] then
          binPrint (fun a b => s!"(uint16_t)gx_f16x2_mul((uint16_t)({a}), (uint16_t)({b}))")
        else if toks == ["mul", "bf16x2"] then
          binPrint (fun a b => CTemplate.mulBF16x2.print a b "")
        -- Scalar mul{.rn}.bf16 is the low lane of the proved packed lane model
        -- (independent lanes; the zero high lane cannot affect the low lane).
        else if toks == ["mul", "bf16"] || toks == ["mul", "rn", "bf16"] then
          binPrint (fun a b =>
            s!"(uint16_t)({CTemplate.mulBF16x2.print s!"(uint32_t)(uint16_t)({a})" s!"(uint32_t)(uint16_t)({b})" ""})")
        else if rawFloat then bin .mul
        else if toks.contains "f32" || toks.contains "f64" then floatArith.getD trapOp
        else
          match scalarArithTy? tyTok with
          | some (sgn, w) =>
            match toks with
            | ["mul", mode, parsedTy] =>
             if parsedTy != tyTok then trapOp
             else if mode == "lo" then bin .mul
             else if mode == "hi" then
              match args with
              | [d, a, b] =>
                let e := match w with
                  | .w16 => printMulHi16 sgn (oC a) (oC b)
                  | .w32 => printMulHi sgn .w32 (oC a) (oC b)
                  | .w64 => printMulHi sgn .w64 (oC a) (oC b)
                  | .w8 => ""
                s!"    {dest d} = {e};"
              | _ => trapOp
             else if mode == "wide" then
              match w, args with
              | .w16, [d, a, b] =>
                s!"    {dest d} = {printMulWide sgn false (oC a) (oC b)};"
              | .w32, [d, a, b] =>
                s!"    {dest d} = {printMulWide sgn true (oC a) (oC b)};"
              | _, _ => trapOp
             else trapOp
            | _ => trapOp
          | none => floatArith.getD trapOp
      /- PTX definition: integer `div` interprets operands using the declared signedness, while floating forms follow floating arithmetic. -/
      /- Lean modeling: six integer variants select proved models; exact RN f32/f64 and full-range approximate f32 retain trusted IEEE division, while every other float mode traps. -/
      | "div"  =>
        let rawFloat := toks == ["div", "rn", "f32"] || toks == ["div", "rn", "f64"]
        if toks == ["div", "full", "f32"] then
          binPrint (fun a b => s!"gx_div_full_f32({a}, {b})")
        -- PTX defines div.approx.f32 as d = a * (1/b) (hardware reciprocal is
        -- approximate, max 2 ulp); modelled with an RN reciprocal then RN product.
        else if toks == ["div", "approx", "f32"] || toks == ["div", "approx", "ftz", "f32"] then
          binPrint (fun a b => s!"gx_div_approx_f32({a}, {b}, {if toks.contains "ftz" then "true" else "false"})")
        else if tyTok == "u16" && toks.length == 2 then binPrint printDivU16
        else if tyTok == "s16" && toks.length == 2 then binPrint printDivS16
        else if plainScalarInt || rawFloat then bin (emitDiv ty)
        else trapOp
      /- PTX definition: `rem` returns the integer remainder under the declared signed or unsigned interpretation. -/
      /- Lean modeling: six named integer models use GX's fixed-width casts; negative/zero machine policy and printing remain explicit boundaries. -/
      | "rem"  =>
        if tyTok == "u16" && toks.length == 2 then binPrint printRemU16
        else if tyTok == "s16" && toks.length == 2 then binPrint printRemS16
        else if plainScalarInt then bin (emitRem ty)
        else trapOp
      /- PTX definition: `min` returns the lesser operand under its declared numeric interpretation. -/
      /- Lean modeling: integer variants use named models; plain and xorsign.abs f32 use a tested encoding helper, f64 retains host fmin, and other modifier/third-source forms trap. -/
      | "min"  =>
        if !simdTy.isEmpty then simdRoute.getD trapOp else
        if tyTok == "u16" && toks.length == 2 then binPrint printMinU16
        else if tyTok == "s16" && toks.length == 2 then binPrint printMinS16
        else if plainScalarInt then bin (emitMin ty)
        else if [["min", "f32"], ["min", "NaN", "f32"], ["min", "xorsign", "abs", "f32"],
                  ["min", "ftz", "f32"], ["min", "ftz", "NaN", "f32"]].contains toks then
          -- .ftz flushes subnormal sources and the result (sign preserved).
          match args with
          | [d, a, b] =>
            if d == "_" then trapOp else
              let ftz := toks.contains "ftz"
              let src (x : String) := if ftz then s!"gx_ftz_f32_bits({floatBitsOperand regs "f32" x})" else floatBitsOperand regs "f32" x
              let r := s!"gx_minmax_f32_bits({src a}, {src b}, false, {if toks.contains "NaN" then "true" else "false"}, {if toks.contains "xorsign" then "true" else "false"})"
              fWrite d s!"gx_bits_to_f32({if ftz then s!"gx_ftz_f32_bits({r})" else r})"
          | _ => trapOp
        else if toks == ["min", "f64"] then binPrint (fun a b => s!"fmin({a}, {b})")
        else trapOp
      /- PTX definition: `max` returns the greater operand under its declared numeric interpretation. -/
      /- Lean modeling: scalar integer variants use named models and s32 ReLU composes two signed maxima; plain/NaN f32 use a tested encoding helper, f64 retains host fmax, and other forms trap. -/
      | "max"  =>
        if !simdTy.isEmpty then simdRoute.getD trapOp else
        if tyTok == "u16" && toks.length == 2 then binPrint printMaxU16
        else if tyTok == "s16" && toks.length == 2 then binPrint printMaxS16
        else if plainScalarInt then bin (emitMax ty)
        else if toks == ["max", "relu", "s32"] then
          match args with
          | [d, a, b] =>
            if d == "_" then trapOp else
              let v := (CTemplate.smax .w32).print (oC a) (oC b) ""
              s!"    {dest d} = {(CTemplate.smax .w32).print v "0" ""};"
          | _ => trapOp
        else if [["max", "f32"], ["max", "NaN", "f32"], ["max", "ftz", "f32"], ["max", "ftz", "NaN", "f32"]].contains toks then
          match args with
          | [d, a, b] =>
            if d == "_" then trapOp else
              let ftz := toks.contains "ftz"
              let src (x : String) := if ftz then s!"gx_ftz_f32_bits({floatBitsOperand regs "f32" x})" else floatBitsOperand regs "f32" x
              let r := s!"gx_minmax_f32_bits({src a}, {src b}, true, {if toks.contains "NaN" then "true" else "false"}, false)"
              fWrite d s!"gx_bits_to_f32({if ftz then s!"gx_ftz_f32_bits({r})" else r})"
          | _ => trapOp
        else if toks == ["max", "f64"] then binPrint (fun a b => s!"fmax({a}, {b})")
        else trapOp
      /- PTX definition: `mad` adds a third operand to the selected low, high, or wide product form. -/
      /- Lean modeling: all 16 non-saturating integer variants select named low/high/wide models; float, carry, and saturation forms trap. -/
      | "mad"  =>
        match scalarArithTy? tyTok with
          | some (sgn, w) =>
            match toks with
            | ["mad", mode, parsedTy] =>
             if parsedTy != tyTok then trapOp
             else
              match args with
              | [d, a, b, c] =>
                if mode == "lo" then
                  s!"    {dest d} = {(CExpr.mad (operand a) (operand b) (operand c)).toC k};"
                else if mode == "hi" then
                  let e := match w with
                    | .w16 => printMadHi16 sgn (oC a) (oC b) (oC c)
                    | .w32 => printMadHi sgn .w32 (oC a) (oC b) (oC c)
                    | .w64 => printMadHi sgn .w64 (oC a) (oC b) (oC c)
                    | .w8 => ""
                  s!"    {dest d} = {e};"
                else if mode == "wide" then
                  match w with
                  | .w16 => s!"    {dest d} = {printMadWide sgn false (oC a) (oC b) (oC c)};"
                  | .w32 => s!"    {dest d} = {printMadWide sgn true (oC a) (oC b) (oC c)};"
                  | _ => trapOp
                else trapOp
              | _ => trapOp
            | _ => trapOp
          | none => trapOp
      /- PTX definition: `fma` computes a fused floating multiply-add with the selected precision and modifiers. -/
      /- Rounding (.rz/.rm/.rp), .ftz and .sat forms of f32/f64, `.f32x2` and mixed `.f32.{f16,bf16}` route through `gx_farith32/64`, which set the host rounding mode around a single IEEE operation (`trusted:template-print`). -/
      /- Lean modeling: exact RN f32/f64 selects `fmaf`/`fma` and RN-FTZ f32 flushes inputs/results around fused `fmaf`; host fusion/rounding is trusted and every other modifier/format traps. -/
      | "fma"  =>
        if toks == ["fma", "rn", "ftz", "f32"] then
          match args with
          | [d, a, b, c] =>
            if d == "_" then trapOp
            else fWrite d s!"gx_fma_rn_ftz_f32({fC a}, {fC b}, {fC c})"
          | _ => trapOp
        else if toks == ["fma", "rn", "f32"] || toks == ["fma", "rn", "f64"] then
          match args with
          | [d, a, b, c] =>
            let fn := if tyTok == "f64" then "fma" else "fmaf"
            if d == "_" then trapOp else fWrite d s!"{fn}({fC a}, {fC b}, {fC c})"
          | _ => trapOp
        else floatArith.getD trapOp
      /- PTX definition: `ex2.approx` computes an approximate base-two exponential for a floating operand. -/
      /- Lean modeling: exact approximate f32 selects trusted `exp2f`, optionally with explicit signed FTZ inputs/results; every other type/modifier still traps. -/
      | "ex2"  =>
        if toks == ["ex2", "approx", "ftz", "f32"] then
          match args with
          | [d, a] => if d == "_" then trapOp else fWrite d s!"gx_ex2_approx_ftz_f32({fC a})"
          | _ => trapOp
        else if toks == ["ex2", "approx", "f32"] then
          match args with
          | [d, a] => if d == "_" then trapOp else fWrite d s!"exp2f({fC a})"
          | _ => trapOp
        else trapOp
      /- PTX definition: `lg2.approx` computes an approximate base-two logarithm for a floating operand. -/
      /- Lean modeling: only exact `lg2.approx.f32` selects trusted `log2f`; FTZ and every other spelling trap until modeled. -/
      | "lg2"  =>
        -- .ftz: subnormal source flushes to zero (log2 -> -inf), result flushed.
        if toks == ["lg2", "approx", "ftz", "f32"] then
          match args with
          | [d, a] => if d == "_" then trapOp else fWrite d s!"gx_ftz_f32(log2f(gx_ftz_f32({fC a})))"
          | _ => trapOp
        else if toks == ["lg2", "approx", "f32"] then
          match args with
          | [d, a] => if d == "_" then trapOp else fWrite d s!"log2f({fC a})"
          | _ => trapOp
        else trapOp
      /- PTX definition: `rcp` computes a reciprocal under the selected approximation/rounding form. -/
      /- Lean modeling: only exact RN f32/f64 syntax selects trusted host division; approximation, FTZ, half/bfloat, and directed-rounding forms trap. -/
      | "rcp"  =>
        if toks == ["rcp", "approx", "ftz", "f32"] then
          match args with
          | [d, a] => if d == "_" then trapOp else fWrite d s!"gx_rcp_approx_ftz_f32({fC a})"
          | _ => trapOp
        else if toks == ["rcp", "rn", "f32"] || toks == ["rcp", "rn", "f64"] then
          match args with
          | [d, a] => if d == "_" then trapOp else fWrite d s!"(({ctypeOf tyTok})1 / ({fC a}))"
          | _ => trapOp
        else trapOp
      /- PTX definition: `rsqrt.approx.ftz.f32` approximates reciprocal square root and flushes subnormal inputs/results to sign-preserving zero. -/
      /- Lean modeling: this exact syntax calls a bit-FTZ and trusted IEEE/libm helper; approximation bounds are numerically tested, not Lean theorems, and the non-FTZ f32 form preserves subnormals; other rsqrt forms trap. -/
      | "rsqrt" =>
        if toks == ["rsqrt", "approx", "ftz", "f32"] then
          match args with
          | [d, a] => if d == "_" then trapOp else fWrite d s!"gx_rsqrt_approx_ftz_f32({fC a})"
          | _ => trapOp
        else if toks == ["rsqrt", "approx", "f32"] then
          match args with
          | [d, a] => if d == "_" then trapOp else fWrite d s!"gx_rsqrt_approx_f32({fC a})"
          | _ => trapOp
        else trapOp
      /- PTX definition: sqrt.approx.ftz.f32 computes square root with sign-preserving input/output subnormal flushing. -/
      /- Lean modeling: this form and sqrt.rn.f32 use trusted IEEE/libm helpers; numerical tests check special values and approximation bounds, not hardware bit identity. -/
      | "sqrt" =>
        if toks == ["sqrt", "approx", "ftz", "f32"] then
          match args with
          | [d, a] => if d == "_" then trapOp else fWrite d s!"gx_sqrt_approx_ftz_f32({fC a})"
          | _ => trapOp
        else if toks == ["sqrt", "rn", "f32"] then
          match args with
          | [d, a] => if d == "_" then trapOp else fWrite d s!"gx_sqrt_rn_f32({fC a})"
          | _ => trapOp
        else trapOp
      /- PTX definition: `copysign.{f32,f64} d, a, b` returns b with its sign bit replaced by the sign bit of a (PTX ISA 9.7.3.2); NaN payloads pass through unchanged. -/
      /- Lean modeling: an exact-syntax bit transcription on the raw IEEE encodings (sign mask from a, magnitude bits from b); register bit access and printing are trusted. -/
      | "copysign" =>
        let args := args.map (fun a => strim (if a.endsWith ";" then a.dropRight 1 else a))
        match toks, args with
        | ["copysign", t], [d, a, b] =>
          if d == "_" || !(t == "f32" || t == "f64") then trapOp
          else
            let (ut, sign) := if t == "f32" then ("uint32_t", "0x80000000u") else ("uint64_t", "0x8000000000000000ull")
            let ab := floatBitsOperand regs t a
            let bb := floatBitsOperand regs t b
            s!"    gx_movb({dest d}, ({ut})(((({ut})({bb})) & ~({ut}){sign}) | ((({ut})({ab})) & ({ut}){sign})));"
        | _, _ => trapOp
      /- PTX definition: integer `neg` forms the arithmetic negation of its source at the declared type. -/
      /- Lean modeling: three signed integers use named modular templates; floating forms use trusted sign-bit helpers (gx_absneg_*) and all other forms trap. -/
      | "neg"  =>
        match scalarArithTy? tyTok with
        | some (true, _) =>
          if toks.length == 2 then
            match args with
            | [d, a] => s!"    {dest d} = {(CExpr.neg (operand a)).toC k};"
            | _ => trapOp
          else trapOp
        | _ =>
          -- Floating neg: same type set as abs; flips the sign bit.
          match args with
          | [d, a] =>
            let mods := toks.drop 1
            let ftz := if mods.contains "ftz" then "1" else "0"
            if d == "_" then trapOp
            else if mods == ["f32"] || mods == ["ftz", "f32"] then
              fWrite d s!"gx_absneg_f32(1, {ftz}, {fC a})"
            else if mods == ["f64"] then fWrite d s!"gx_absneg_f64(1, {fC a})"
            else if [["f16"], ["ftz", "f16"], ["f16x2"], ["ftz", "f16x2"], ["bf16"], ["bf16x2"]].contains mods then
              let hty := mods.getLastD ""
              let lanes := if hty.endsWith "x2" then "2" else "1"
              let cast := if lanes == "2" then "uint32_t" else "uint16_t"
              let bf := if hty.startsWith "bf" then "1" else "0"
              s!"    {dest d} = ({cast})gx_absneg_h16(1, {ftz}, {bf}, {lanes}, (uint32_t)({oC a}));"
            else trapOp
          | _ => trapOp
      -- ==================================================================
      -- LOGIC AND COMPARISON
      -- ==================================================================
      /- PTX definition: `shl` shifts left and produces zero when the shift amount is at least the operand width. -/
      /- Lean modeling: the shallow shift-template value equality backs modeled widths; helper printing is trusted and subword certificate coverage remains an explicit gap. -/
      | "shl"  => bin .shl
      /- PTX definition: signed `shr` sign-fills and bit/unsigned `shr` zero-fills with width-limited shift amounts. -/
      /- Lean modeling: `emitShr` selects shallow value-equal templates; the generated helper/compiler link is trusted and subword certificate coverage remains gapped. -/
      | "shr"  => bin (emitShr ty)          -- verified signed/unsigned split
      /- PTX definition: `and` combines either predicate truth values or same-width bit operands. -/
      /- Lean modeling: operator selection and template values have shallow equalities; rendered C remains trusted and 8/16-bit integer certificates remain explicitly gapped. -/
      | "and"  => bin (emitBin ty .and)     -- verified pred/int split
      /- PTX definition: `or` combines either predicate truth values or same-width bit operands. -/
      /- Lean modeling: operator selection and template values have shallow equalities; rendered C remains trusted and 8/16-bit integer certificates remain explicitly gapped. -/
      | "or"   => bin (emitBin ty .or)
      /- PTX definition: `xor` combines either predicate truth values or same-width bit operands. -/
      /- Lean modeling: operator selection and template values have shallow equalities; rendered C remains trusted and 8/16-bit integer certificates remain explicitly gapped. -/
      | "xor"  => bin (emitBin ty .xor)
      /- PTX definition: `not` complements a predicate or each bit of a same-width bit operand. -/
      /- Lean modeling: operator selection and template values have shallow equalities; rendered C remains trusted and 8/16-bit integer certificates remain explicitly gapped. -/
      | "not"  => match args with
                  | d :: a :: _ => s!"    {dest d} = {(emitNot ty (operand a)).toC k};"
                  | _ => s!"    /* malformed: {s} */"
      /- PTX definition: `setp` compares typed operands and writes the resulting predicate, with ordered and unordered floating variants. -/
      /- Lean modeling: integer comparisons use the shallow `CTemplate.cmp` model; f32/f64 use a tested IEEE-encoding comparator with ordered/unordered NaN rules, while dual destinations, BoolOp, and sinks still trap. -/
      | "setp" =>
        -- cast both operands to the comparison type so signedness AND width
        -- are correct (`setp.lt.s32` must be a SIGNED 32-bit `<`). Integer
        -- comparisons use a shallow-value-backed `CTemplate.cmp`; its printed
        -- spelling/compiler link remains `trusted:template-print`.
        -- Float comparisons have a separate bit-based helper: ordered `ne`
        -- must reject NaN, unlike native C !=. BoolOp, p|q and f16/bf16(x2)
        -- forms take the general route below, which evaluates the simple
        -- comparison (or gx_setp_half) and combines it with the predicate.
        -- For exact b32/b64 eq/ne, the actual declared C type drives bit
        -- reinterpretation, independent of register-name prefixes.
        -- General forms (PTX ISA 9.3 §9.7.3.5, §9.7.7.2): optional BoolOp with
        -- a third predicate, p|q destinations, and f16/bf16(x2) operands.
        let boolOp := (toks.drop 2).filter (fun t => ["and", "or", "xor"].contains t)
        let halfTy := toks.getLastD "" == "f16" || toks.getLastD "" == "bf16" ||
                      toks.getLastD "" == "f16x2" || toks.getLastD "" == "bf16x2"
        let dests := (args.headD "").splitOn "|"
        let general := !boolOp.isEmpty || dests.length == 2 || halfTy
        if general then
          let ty := toks.getLastD ""
          let cmp := toks.getD 1 ""
          let ftz := toks.contains "ftz"
          let combine (t : String) : String :=
            match boolOp, args.getD 3 "" with
            | [], _ => t
            | [bo], c =>
              let neg := c.startsWith "!"
              let cv := if neg then s!"(!{oC (c.drop 1)})" else s!"({oC c})"
              let opc := if bo == "and" then "&&" else if bo == "or" then "||" else "!="
              s!"(({t}) {opc} {cv})"
            | _, _ => ""
          let shapeOK := toks.length == 3 + boolOp.length + (if ftz then 1 else 0) &&
            (boolOp.isEmpty || args.length == 4) && (!boolOp.isEmpty || args.length == 3) &&
            dests.all (fun x => x != "_" && !x.isEmpty) && dests.length ≤ 2 &&
            (!ftz || ty == "f32" || ty == "f16" || ty == "f16x2")
          if !shapeOK then trapOp
          else if ty == "f16" || ty == "bf16" || ty == "f16x2" || ty == "bf16x2" then
            match floatCmpCode? cmp with
            | none => trapOp
            | some code =>
              let bf := if ty.startsWith "bf16" then "true" else "false"
              let half (x : String) (i : Nat) := s!"(uint16_t)(((uint32_t)({oC x})) >> {16 * i})"
              let t (i : Nat) := s!"gx_setp_half({half (args.getD 1 "") i}, {half (args.getD 2 "") i}, {code}u, {bf}, {if ftz then "true" else "false"})"
              if ty.endsWith "x2" then
                if dests.length != 2 then trapOp
                else s!"    {dest (dests.headD "")} = {combine (t 0)}; {dest (dests.getD 1 "")} = {combine (t 1)};"
              else if dests.length == 2 then
                "    { int _gx_t = " ++ t 0 ++ "; " ++ dest (dests.headD "") ++ " = " ++ combine "_gx_t" ++ "; " ++
                  dest (dests.getD 1 "") ++ " = " ++ combine "!_gx_t" ++ "; }"
              else s!"    {dest (dests.headD "")} = {combine (t 0)};"
          else
            -- Integer/f32/f64: evaluate the simple comparison into a temporary.
            let simpleOp := String.intercalate "." (["setp", cmp] ++ (if ftz then ["ftz"] else []) ++ [ty])
            let inner := strim (translateStmtScoped s!"{simpleOp} %gx_setp_t, {args.getD 1 ""}, {args.getD 2 ""}" guardGlobal (profile := false))
            if (inner.splitOn "gx_ptx_trap").length > 1 then trapOp
            else if dests.length == 2 then
              "    { int R_gx_setp_t; " ++ inner ++ " " ++ dest (dests.headD "") ++ " = " ++ combine "R_gx_setp_t" ++ "; " ++
                dest (dests.getD 1 "") ++ " = " ++ combine "!R_gx_setp_t" ++ "; }"
            else "    { int R_gx_setp_t; " ++ inner ++ " " ++ dest (dests.headD "") ++ " = " ++ combine "R_gx_setp_t" ++ "; }"
        else
        match args, toks with
        | [d, a, b], ["setp", cmp, "ftz", "f32"] =>
          (printFloatSetp? regs cmp "f32" d a b true).getD trapOp
        | [d, a, b], ["setp", cmp, parsedTy] =>
          if parsedTy == "f32" || parsedTy == "f64" then
            (printFloatSetp? regs cmp parsedTy d a b).getD trapOp
          else
          let intTypes := ["b16", "b32", "b64", "u16", "u32", "u64",
                           "s16", "s32", "s64"]
          let cmpLegal :=
            if parsedTy.startsWith "b" then cmp == "eq" || cmp == "ne"
            else if parsedTy.startsWith "u" then
              ["eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs"].contains cmp
            else ["eq", "ne", "lt", "le", "gt", "ge"].contains cmp
          if d == "_" || (d.splitOn "|").length != 1 || !intTypes.contains parsedTy || !cmpLegal then trapOp
          else
            match cvtIntTy? parsedTy, cmpOp? cmp with
            | some (sgn, .w16), some cop =>
              let e := printCmp16 sgn cop (oC a) (oC b)
              s!"    {dest d} = {e};"
            | some (sgn, .w32), some cop =>
              let bits (x : String) :=
                if parsedTy == "b32" then bitOperand regs "b32" x
                else oC x
              let e := (CTemplate.cmp sgn cop .w32).print (bits a) (bits b) ""
              s!"    {dest d} = {e};"
            | some (sgn, .w64), some cop =>
              let bits (x : String) :=
                if parsedTy == "b64" then bitOperand regs "b64" x
                else oC x
              let e := (CTemplate.cmp sgn cop .w64).print (bits a) (bits b) ""
              s!"    {dest d} = {e};"
            | _, _ => trapOp
        | _, _ => trapOp
      /- PTX definition: `selp` selects its first value when the predicate is true and its second value otherwise. -/
      /- Lean modeling: the selector/template has shallow value equality; the emitted conditional spelling and compiler interpretation remain `trusted:template-print`. -/
      | "selp" => match args with
                  | [d, a, b, p] =>
                    if toks.length != 2 then trapOp
                    else if tyTok == "f32" || tyTok == "f64" then
                      if d == "_" then trapOp
                      else fWrite d ((CExpr.sel (operand p) (.var (fC a)) (.var (fC b))).toC .f)
                    else
                      let valid := ["b16", "b32", "b64", "u16", "u32", "u64", "s16", "s32", "s64"]
                      if toks.length != 2 || !valid.contains tyTok || args.length != 4 || d == "_" then trapOp
                      else
                        -- Type both branches before C's usual arithmetic
                        -- conversions. In particular, -1 : 1u must not become
                        -- unsigned32 before assignment to a 64-bit register.
                        let value (x : String) := if tyTok.startsWith "b" then bitOperand regs tyTok x else s!"({ctypeOf tyTok})({oC x})"
                        let e := s!"({oC p} ? {value a} : {value b})"
                        if tyTok.startsWith "b" then s!"    gx_movb({dest d}, ({ctypeOf tyTok})({e}));"
                        else s!"    {dest d} = {e};"
                  | _ => trapOp
      -- ==================================================================
      -- MEMORY AND CONVERSION
      -- ==================================================================
      /- PTX definition: `mma.sync.aligned` performs a warp-collective D = A*B + C on register fragments laid out per shape and type. -/
      /- Lean modeling: the m16n8k8/k16 f16-bf16(-tf32) and m16n8k32 e4m3/e5m2 row.col forms with f32/f16 C/D route to a trusted full-warp helper; operand counts are checked against the shape; other shapes, integer and block-scaled forms trap. -/
      | "mma" =>
        let fmt (t : String) : Option Nat := match t with
          | "f16" => some 0 | "bf16" => some 1 | "e4m3" => some 2 | "e5m2" => some 3 | "tf32" => some 4 | _ => none
        let intToks := toks.filter (· != "satfinite")
        let intRoute : Option String :=
          match intToks with
          | ["mma", "sync", "aligned", shape, "row", "col", "s32", aty, bty, "s32"] =>
            let ik := if shape == "m16n8k16" then 16 else if shape == "m16n8k32" then 32 else 0
            let na := ik / 8
            let nb := ik / 16
            let modifiersOK := toks == intToks || toks == (intToks.take 6 ++ ["satfinite"] ++ intToks.drop 6)
            if ik == 0 || !["s8", "u8"].contains aty || !["s8", "u8"].contains bty ||
               !modifiersOK || args.length != 8 + na + nb then none
            else
              let ds := args.take 4
              let vals (xs : List String) := String.intercalate ", " (xs.map (fun x => s!"gx_bits32({oC x})"))
              if ds.any (· == "_") then none else
                let writes := String.intercalate " " (ds.enum.map (fun (i,r) => s!"gx_movb({dest r}, _gx_id[{i}]);"))
                some ("    { const uint32_t _gx_ia[4] = {" ++ vals ((args.drop 4).take na) ++
                  "}, _gx_ib[2] = {" ++ vals ((args.drop (4+na)).take nb) ++ "}, _gx_ic[4] = {" ++ vals (args.drop (4+na+nb)) ++
                  "}; uint32_t _gx_id[4]; " ++ s!"gx_mma_sync_i8({ik}u, {if aty == "s8" then "true" else "false"}, {if bty == "s8" then "true" else "false"}, {if toks.contains "satfinite" then "true" else "false"}, " ++
                  "_gx_ia, _gx_ib, _gx_ic, _gx_id); " ++ writes ++ " }")
          | _ => none
        if intRoute.isSome then intRoute.getD trapOp else
        match toks with
        | ["mma", "sync", "aligned", shape, "row", "col", dty, aty, bfy, cty] =>
          let k := if shape == "m16n8k8" then 8 else if shape == "m16n8k16" then 16 else if shape == "m16n8k32" then 32 else 0
          match fmt aty, fmt bfy with
          | some af, some bf =>
            let okTypes := (dty == "f32" || dty == "f16") && (cty == "f32" || cty == "f16") &&
              ((k == 8 && af == bf && (af ≤ 1 || af == 4)) || (k == 16 && af == bf && af ≤ 1) ||
               (k == 32 && (af == 2 || af == 3) && (bf == 2 || bf == 3))) &&
              (af != 4 || (dty == "f32" && cty == "f32")) && (af != 1 || (dty == "f32" && cty == "f32"))
            let na := if k == 8 && af != 4 then 2 else 4
            let nb := if k == 8 && af != 4 then 1 else 2
            let nd := if dty == "f32" then 4 else 2
            let nc := if cty == "f32" then 4 else 2
            if !okTypes || args.length != nd + na + nb + nc then trapOp
            else
              let ds := args.take nd
              let as_ := (args.drop nd).take na
              let bs := (args.drop (nd + na)).take nb
              let cs := args.drop (nd + na + nb)
              let vals (xs : List String) := String.intercalate ", " (xs.map (fun x => s!"gx_bits32({oC x})"))
              if ds.any (· == "_") then trapOp
              else
                let writes := String.intercalate " " (ds.enum.map (fun (i, r) => s!"gx_movb({dest r}, _gx_md[{i}]);"))
                "    { const uint32_t _gx_ma[4] = {" ++ vals as_ ++ "}, _gx_mb[2] = {" ++ vals bs ++ "}, _gx_mc[4] = {" ++ vals cs ++
                  "}; uint32_t _gx_md[4]; gx_mma_sync(" ++ toString k ++ "u, " ++ toString af ++ "u, " ++ toString bf ++ "u, " ++
                  (if cty == "f32" then "1u" else "0u") ++ ", " ++ (if dty == "f32" then "1u" else "0u") ++
                  ", _gx_ma, _gx_mb, _gx_mc, _gx_md); " ++ writes ++ " }"
          | _, _ => trapOp
        | _ => trapOp
      /- PTX definition: aligned WMMA loads/stores fragments and computes D=A*B+C collectively across a full warp. -/
      /- Lean modeling: f16 m16n16k16 and bf16 m16n16k16/m8n32k16/m32n8k16 with f32 accumulators use consistent emulator fragment layouts and trusted warp rendezvous; other formats/modifiers trap. -/
      | "wmma" =>
        let args := args.map (fun a => strim (if a.endsWith ";" then a.dropRight 1 else a))
        let bf := toks.contains "bf16"
        let (m, n) : Nat × Nat := if toks.contains "m8n32k16" then (8,32) else if toks.contains "m32n8k16" then (32,8) else (16,16)
        let shape := s!"m{m}n{n}k16"
        let na := if bf then m / 4 else 8
        let nb := if bf then n / 4 else 8
        let input := if bf then "bf16" else "f16"
        let loadOps := (["a", "b"].flatMap fun role => ["row", "col"].flatMap fun layout =>
          ["", ".shared", ".global"].map fun space => s!"wmma.load.{role}.sync.aligned.{layout}.{shape}{space}.{input}")
        let accLoadOps := (["row", "col"].flatMap fun layout =>
          ["", ".shared", ".global"].map fun space => s!"wmma.load.c.sync.aligned.{layout}.{shape}{space}.f32")
        let storeOps := (["row", "col"].flatMap fun layout =>
          ["", ".shared", ".global"].map fun space => s!"wmma.store.d.sync.aligned.{layout}.{shape}{space}.f32")
        let mmaTypes := if bf then "f32.bf16.bf16.f32" else "f32.f32"
        let mmaOps := (["row", "col"].flatMap fun al => ["row", "col"].map fun bl =>
          s!"wmma.mma.sync.aligned.{al}.{bl}.{shape}.{mmaTypes}")
        let memSpace := if isShared then 2 else if toks.contains "global" then 1 else 0
        let column := if toks.contains "col" then "true" else "false"
        let guard := if guarded then "true" else "false"
        let uvals (xs : List String) := String.intercalate ", " (xs.map (fun x => s!"(uint32_t)({oC x})"))
        let fvals (xs : List String) := String.intercalate ", " (xs.map (fun x => s!"gx_wmma_f32({oC x})"))
        let writes (xs : List String) := String.intercalate " " (xs.enum.map (fun (i,d) => s!"gx_movb({dest d}, _gx_wd[{i}]);"))
        let validD (count : Nat) (xs : List String) := xs.length == count && xs.eraseDups.length == count && xs.all (fun d => d != "_" && (d.startsWith "%" || regs.contains d))
        let isA := toks.contains "a"
        let nr := if isA then na else nb
        let rows := if isA then m else 16
        let cols := if isA then 16 else n
        let stride (idx fallback : Nat) := if args.length > idx then oC (args.getD idx "") else toString fallback
        if !wmmaVector then trapOp
        else if loadOps.contains op && (bf || (m == 16 && n == 16)) && (args.length == nr + 1 || args.length == nr + 2) && validD nr (args.take nr) then
          "    { uint32_t _gx_wd[" ++ toString nr ++ s!"]; gx_wmma_load_16<{guard}, {column}, {rows}, {cols}, {nr}, {memSpace}>(" ++
            mkAddr (args.getD nr "") ++ ", " ++ stride (nr+1) (if column == "true" then rows else cols) ++ ", _gx_wd); " ++ writes (args.take nr) ++ " }"
        else if accLoadOps.contains op && (args.length == 9 || args.length == 10) && validD 8 (args.take 8) then
          "    { float _gx_wd[8]; " ++ s!"gx_wmma_load_f32<{guard}, {column}, {m}, {n}, {memSpace}>(" ++
            mkAddr (args.getD 8 "") ++ ", " ++ stride 9 (if column == "true" then m else n) ++ ", _gx_wd); " ++ writes (args.take 8) ++ " }"
        else if storeOps.contains op && (args.length == 9 || args.length == 10) then
          "    { float _gx_ws[8] = {" ++ fvals ((args.drop 1).take 8) ++ "}; " ++ s!"gx_wmma_store_f32<{guard}, {column}, {m}, {n}, {memSpace}>(" ++
            mkAddr (args.headD "") ++ ", " ++ stride 9 (if column == "true" then m else n) ++ ", _gx_ws); }"
        else if mmaOps.contains op && (bf || (m == 16 && n == 16)) && args.length == 16 + na + nb && validD 8 (args.take 8) then
          "    { uint32_t _gx_wa[" ++ toString na ++ "] = {" ++ uvals ((args.drop 8).take na) ++ "}; uint32_t _gx_wb[" ++ toString nb ++ "] = {" ++
            uvals ((args.drop (8+na)).take nb) ++ "}; float _gx_wc[8] = {" ++ fvals (args.drop (8+na+nb)) ++
            "}; float _gx_wd[8]; " ++ (if bf then s!"gx_wmma_bf16({m}, {n}, " else "gx_wmma_m16n16k16_f16(") ++
            "_gx_wa, _gx_wb, _gx_wc, _gx_wd); " ++ writes (args.take 8) ++ " }"
        else trapOp
      /- PTX definition: wgmma.fence, commit_group and wait_group N are aligned warpgroup collectives; dense wgmma.mma_async.m64nNkK computes D = A*B (+D when scale-d) over f16/bf16 (f16/f32 accumulators, optional MN-major transposes), tf32, e4m3/e5m2 (f16/f32) and s8/u8 (s32, optional .satfinite) with A from a shared-memory descriptor or four fragment registers, B from a descriptor, and imm-scale-a/b negation for floating kinds (PTX ISA 9.7.16.5.2). -/
      /- Lean modeling: the opcode, shape (N and K for the kind), per-kind operand list and vector arities captured before brace normalization are checked here; accepted forms call the trusted eager-completion runtime helper `gx_wgmma_mma_async` (warpgroup rendezvous, descriptor/fragment layouts, numerics) with every accumulator register bit-copied in and out. Sparse, b1 and malformed forms trap. This is not an asynchronous execution or floating-point proof. -/
      | "wgmma" =>
        let op := if op.endsWith ";" then op.dropRight 1 else op
        let toks := op.splitOn "."
        let args := (args.map (fun a => strim (if a.endsWith ";" then a.dropRight 1 else a))).filter (! ·.isEmpty)
        let isReg (x : String) := x != "_" && (x.startsWith "%" || regs.contains x)
        match toks with
        | ["wgmma", "fence", "sync", "aligned"] =>
          if args.isEmpty then "    gx_wgmma_collective(1u, 0u);" else trapOp
        | ["wgmma", "commit_group", "sync", "aligned"] =>
          if args.isEmpty then "    gx_wgmma_collective(2u, 0u);" else trapOp
        | ["wgmma", "wait_group", "sync", "aligned"] =>
          match args with
          | [count] =>
            match count.toNat? with
            | some groups => s!"    gx_wgmma_collective(3u, {groups}u);"
            | none => trapOp
          | _ => trapOp
        | "wgmma" :: "mma_async" :: "sync" :: "aligned" :: shape :: types =>
          let satfinite := types.contains "satfinite"
          let types := types.filter (· != "satfinite")
          -- format codes shared with emu/ptx/wgmma.cc
          let fmt? : String → Option Nat
            | "f16" => some 0 | "bf16" => some 1 | "e4m3" => some 2 | "e5m2" => some 3
            | "tf32" => some 4 | "s8" => some 5 | "u8" => some 6 | _ => none
          let dfmt? : String → Option Nat
            | "f16" => some 0 | "f32" => some 1 | "s32" => some 2 | _ => none
          let dims : Option (Nat × Nat) :=
            match (shape.drop 4).splitOn "k" with
            | [nText, kText] =>
              if shape.startsWith "m64n" then
                match nText.toNat?, kText.toNat? with
                | some n, some kk => some (n, kk)
                | _, _ => none
              else none
            | _ => none
          match types, dims with
          | [dty, aty, bty], some (n, kk) =>
            match dfmt? dty, fmt? aty, fmt? bty with
            | some dt, some af, some bt =>
              let half := af ≤ 1
              let fp8 := af == 2 || af == 3
              let integer := af == 5 || af == 6
              let kindOk :=
                (half && bt == af && (if af == 1 then dt == 1 else dt ≤ 1)) ||
                (af == 4 && bt == 4 && dt == 1) ||
                (fp8 && (bt == 2 || bt == 3) && dt ≤ 1) ||
                (integer && (bt == 5 || bt == 6) && dt == 2)
              let kOk := kk == (if half then 16 else if af == 4 then 8 else 32)
              let nOk := n ≥ 8 && n ≤ 256 && n % 8 == 0 && (!integer || n ≤ 32 || n % 16 == 0)
              let nd := if dt == 0 then n / 4 else n / 2
              match wgmmaShape with
              | some (dCount, aCount) =>
                let regA := aCount == 4
                -- trailing operands after d, a, b: scale-d, then imm-scale-a/b
                -- for floating kinds, then imm-trans-a (descriptor A) and
                -- imm-trans-b for f16/bf16
                let tail := if integer then 1 else if half then (if regA then 4 else 5) else 3
                let aArgs := if regA then 4 else 1
                if !kindOk || !kOk || !nOk || (satfinite && !integer) || dCount != nd ||
                    !(aCount == 0 || regA) || args.length != nd + aArgs + 1 + tail then trapOp
                else
                  let ds := args.take nd
                  let as_ := (args.drop nd).take aArgs
                  let b := args.getD (nd + aArgs) ""
                  let scalars := args.drop (nd + aArgs + 1)
                  let scaleD := scalars.getD 0 ""
                  let imm (i : Nat) := scalars.getD i ""
                  let signOk (x : String) := x == "1" || x == "-1"
                  let transOk (x : String) := x == "0" || x == "1"
                  let scalesOk := integer || (signOk (imm 1) && signOk (imm 2))
                  let transA := half && !regA && imm 3 == "1"
                  let transB := half && imm (if regA then 3 else 4) == "1"
                  let transOkAll := !half || (if regA then transOk (imm 3) else transOk (imm 3) && transOk (imm 4))
                  if ds.any (fun d => !isReg d) || ds.eraseDups.length != nd ||
                      as_.any (fun a => !isReg a) || !isReg b || !scalesOk || !transOkAll ||
                      !(scaleD == "0" || scaleD == "1" || isReg scaleD) then trapOp
                  else
                    let flags := (if !integer && imm 1 == "-1" then 2 else 0) +
                      (if !integer && imm 2 == "-1" then 4 else 0) + (if transA then 8 else 0) +
                      (if transB then 16 else 0) + (if regA then 32 else 0) + (if satfinite then 64 else 0)
                    let initial := String.intercalate ", " (ds.map (fun d =>
                      "(_gx_wf & 1u) ? " ++ bitOperand regs "b32" d ++ " : 0u"))
                    let aVals := String.intercalate ", " (as_.map (fun a => bitOperand regs "b32" a))
                    let writes := String.intercalate " " (ds.enum.map (fun (i, d) =>
                      s!"gx_movb({dest d}, _gx_wd[{i}]);"))
                    let descA := if regA then "0" else bitOperand regs "b64" (as_.headD "")
                    "    { const uint32_t _gx_wf = " ++ toString flags ++ "u | ((" ++ oC scaleD ++
                      ") != 0 ? 1u : 0u); uint32_t _gx_wd[" ++ toString nd ++ "] = {" ++ initial ++
                      "}; const uint32_t _gx_wa[4] = {" ++ (if regA then aVals else "0, 0, 0, 0") ++
                      "}; gx_wgmma_mma_async(" ++ toString n ++ "u, " ++ toString af ++ "u, " ++
                      toString bt ++ "u, " ++ toString dt ++ "u, _gx_wf, " ++ descA ++ ", " ++
                      bitOperand regs "b64" b ++ ", _gx_wa, _gx_wd); " ++ writes ++ " }"
              | none => trapOp
            | _, _, _ => trapOp
          | _, _ => trapOp
        | _ => trapOp
      /- PTX definition: `setmaxnreg.{inc,dec}.sync.aligned.u32 imm-reg-count` hints a new per-thread register maximum for the executing warp; all warps of a warpgroup execute the same instruction, and imm-reg-count is a multiple of 8 in [24, 256] (PTX ISA 9.7.20.5). -/
      /- Lean modeling: the exact syntax and operand range are checked; translated threads have no register file to resize, so the hint has no effect and `.inc` never waits for the CTA pool, while the runtime still rendezvous the warpgroup and traps when its warps execute different setmaxnreg instructions. -/
      | "setmaxnreg" =>
        let args := (args.map (fun a => strim (if a.endsWith ";" then a.dropRight 1 else a))).filter (! ·.isEmpty)
        let opToks := (if op.endsWith ";" then op.dropRight 1 else op).splitOn "."
        match opToks, args with
        | ["setmaxnreg", action, "sync", "aligned", "u32"], [count] =>
          match count.toNat? with
          | some c =>
            if (action == "inc" || action == "dec") && 24 ≤ c && c ≤ 256 && c % 8 == 0 then
              s!"    gx_wgmma_collective({if action == "inc" then 4 else 5}u, {c}u);"
            else trapOp
          | none => trapOp
        | _, _ => trapOp
      /- PTX definition: ldmatrix m8n8.x1/x2/x4 loads matrix fragments, optionally transposed, using warp-supplied row pointers. -/
      /- Lean modeling: explicit shared b16 forms require matching vector arity; fragment mapping and rendezvous are tested trusted runtime behavior, not scalar memory proofs. -/
      | "ldmatrix" =>
        let count := if toks.contains "x1" then 1 else if toks.contains "x2" then 2 else 4
        let trans := toks.contains "trans"
        let expected := s!"ldmatrix.sync.aligned.m8n8.x{count}" ++
          (if trans then ".trans" else "") ++ ".shared.b16"
        -- nvcc/library PTX also places the matrix count before the shape.
        let countFirst := s!"ldmatrix.sync.aligned.x{count}.m8n8" ++
          (if trans then ".trans" else "") ++ ".shared.b16"
        if (op != expected && op != countFirst) || !ldmatrixVector || args.length != count + 1 then trapOp
        else
            let rawAddr := args.getLastD ""
            let a := if rawAddr.endsWith ";" then strim (rawAddr.dropRight 1) else rawAddr
            let dsts := args.take count
            if dsts.any (fun d => d == "_" || !(d.startsWith "%" || regs.contains d)) ||
                !a.startsWith "[" || !a.endsWith "]" then trapOp
            else
              let writes := String.intercalate " " (dsts.enum.map (fun (i, d) =>
                s!"gx_movb({dest d}, _gx_matrix.lane[{i}]);"))
              "    { auto _gx_matrix = gx_ldmatrix_m8n8_b16<" ++ toString count ++ ", " ++
                (if trans then "true" else "false") ++ ">(" ++ mkAddr a ++
                "); " ++ writes ++ " }"
      /- PTX definition: stmatrix m8n8.x1/x2/x4 is the warp-fragment store counterpart to ldmatrix, optionally transposed. -/
      /- Lean modeling: explicit shared b16 forms preserve vector shape; pointer exchange, lane mapping and memory execution are trusted runtime behavior. -/
      | "stmatrix" =>
        let args := args.map (fun a => strim (if a.endsWith ";" then a.dropRight 1 else a))
        let count := if toks.contains "x1" then 1 else if toks.contains "x2" then 2 else 4
        let trans := toks.contains "trans"
        let tail := (if trans then ".trans" else "") ++ ".shared.b16"
        let shapeFirst := s!"stmatrix.sync.aligned.m8n8.x{count}" ++ tail
        let countFirst := s!"stmatrix.sync.aligned.x{count}.m8n8" ++ tail
        if (op != shapeFirst && op != countFirst) || !ldmatrixVector || args.length != count + 1 then trapOp
        else
          let a := args.headD ""
          let vals := args.drop 1
          if !a.startsWith "[" || !a.endsWith "]" ||
              vals.any (fun v => !(v.startsWith "%" || regs.contains v)) then trapOp
          else
            let values := String.intercalate ", " ((List.range 4).map (fun i =>
              if i < count then bitOperand regs "b32" (vals.getD i "0") else "0u"))
            s!"    gx_stmatrix_m8n8_b16<{count}, {if trans then "true" else "false"}>({mkAddr a}, {values});"
      /- PTX definition: `ld`/`ldu` read typed scalar or vector elements from a memory address in the selected state space. -/
      /- Lean modeling: destination arity comes from `.vec`, cache-policy operands no longer displace the address, and type/arity/sink mismatches trap; shared::cluster addresses decode through the `trusted:cluster-runtime` helper; address/guard/offset values are shallow-modeled while the C dereference remains trusted. -/
      | "ld" | "ldu" =>
        -- scalar `[dst, addr]`; vector `[d0,…,dN, addr]` (addr = last arg, an
        -- `[…]` memory operand; braces already stripped so count decides).
        -- Access type AND element stride both come from `mt` (one source of
        -- truth); the element byte offset is the value-model-backed
        -- `CMem.elemOff`. Its shallow address equality does not prove the
        -- printed C dereference/compiler route (`trusted:template-print`).
        -- b32/b64/f32/f64 crossings copy instruction-width bits using the C
        -- local's actual declaration, independent of register-name prefixes.
        let supportedTy := ["b8", "u8", "s8", "b16", "u16", "s16",
                            "b32", "u32", "s32", "b64", "u64", "s64", "f32", "f64"]
        let memTok := toks.getLastD ""
        let vecToks := toks.filter (fun t => t == "v2" || t == "v4" || t == "v8")
        let typeToks := toks.filter (fun t => supportedTy.contains t)
        let sems := ["weak", "volatile", "relaxed", "acquire"]
        let scopes := ["cta", "cluster", "gpu", "sys"]
        let spaces := ["const", "global", "local", "param", "param::entry", "param::func",
                       "shared", "shared::cta", "shared::cluster"]
        let cacheOps := ["ca", "cg", "cs", "lu", "cv", "nc"]
        let l1Hints := ["L1::evict_normal", "L1::evict_unchanged", "L1::evict_first",
                        "L1::evict_last", "L1::no_allocate"]
        let l2Hints := ["L2::evict_normal", "L2::evict_first", "L2::evict_last",
                        "L2::cache_hint", "L2::64B", "L2::128B", "L2::256B"]
        let q := (toks.drop 1).take (toks.length - 2)
        let semQ := q.filter (fun t => sems.contains t)
        let scopeQ := q.filter (fun t => scopes.contains t)
        let spaceQ := q.filter (fun t => spaces.contains t)
        let cacheQ := q.filter (fun t => cacheOps.contains t)
        let l1Q := q.filter (fun t => l1Hints.contains t)
        -- PTX ld grammar has separate optional L2 slots: eviction priority,
        -- cache_hint and prefetch_size (e.g. .L2::evict_normal.L2::256B).
        let l2Prefetch := ["L2::64B", "L2::128B", "L2::256B"]
        let l2EvictQ := q.filter (fun t => l2Hints.contains t && !l2Prefetch.contains t && t != "L2::cache_hint")
        let l2HintQ := q.filter (· == "L2::cache_hint")
        let l2SizeQ := q.filter (fun t => l2Prefetch.contains t)
        let vecQ := q.filter (fun t => t == "v2" || t == "v4" || t == "v8")
        let semScopeOK :=
          if toks.contains "relaxed" || toks.contains "acquire" then scopeQ.length == 1
          else scopeQ.isEmpty
        let syntaxOK :=
          if head == "ldu" then
            q.all (fun t => t == "global" || t == "v2" || t == "v4") &&
              q == (q.filter (· == "global") ++ q.filter (fun t => t == "v2" || t == "v4")) &&
              (q.filter (· == "global")).length ≤ 1
          else
            q.all (fun t => sems.contains t || scopes.contains t || spaces.contains t ||
              cacheOps.contains t || l1Hints.contains t || l2Hints.contains t ||
              t == "v2" || t == "v4" || t == "v8") &&
            semQ.length ≤ 1 && scopeQ.length ≤ 1 && spaceQ.length ≤ 1 && cacheQ.length ≤ 1 &&
            -- PTX §9.7.9.8 examples permute space/sem/scope and even L2/L1
            -- policy order. Validate category membership and uniqueness without
            -- imposing a canonical order absent from the documented examples.
            l1Q.length ≤ 1 && l2EvictQ.length ≤ 1 && l2HintQ.length ≤ 1 &&
              l2SizeQ.length ≤ 1 && vecQ.length ≤ 1 && semScopeOK
        let n := if toks.contains "v8" then 8 else if toks.contains "v4" then 4
                 else if toks.contains "v2" then 2 else 1
        let expectedArgs := n + 1 + (if head == "ld" && toks.contains "L2::cache_hint" then 1 else 0)
        if memTok == "b128" then
          -- 128-bit scalar loads from shared memory (e.g. clusterlaunchcontrol
          -- responses) into an `unsigned __int128` register.
          match args with
          | [d, a] =>
            if isShared && vecToks.isEmpty && d != "_" then
              s!"    memcpy(&{dest d}, (const void*)(uintptr_t)({mkAddr a}), 16);"
            else trapOp
          | _ => trapOp
        else if !syntaxOK || vecToks.length > 1 || typeToks.length != 1 ||
            !supportedTy.contains memTok || args.length != expectedArgs then trapOp
        else
          let mt := memTyOf memTok
          let dsts := args.take n
          if dsts.contains "_" then trapOp
          else
            let rawAddr := args.getD n ""
            -- `.unified` qualifies the address operand; it is not part of the
            -- register/symbol spelling passed to the address renderer.
            let addr := if rawAddr.endsWith ".unified" then rawAddr.dropRight 8 else rawAddr
            let base := mkAddr addr
            let writeLoaded (r value : String) : String :=
              if ["b32", "b64", "f32", "f64"].contains memTok then
                s!"gx_movb({dest r}, ({mt.cname})({value}));"
              else s!"{dest r} = {value};"
            if deviceAddress && toks.contains "volatile" && n == 4 && memTok == "u32" then
              let assigns := String.intercalate " " (dsts.enum.map (fun (i, r) =>
                s!"{dest r} = _gx_v[{i}];"))
              "    { gx_u32x4 _gx_v = gx_gld_v4_u32(" ++ base ++ "); " ++ assigns ++ " }"
            else if n == 1 then
              let a := s!"({base})"
              let r := dsts.headD ""
              s!"    {writeLoaded r (loadE mt.cname a)}"
            else if guarded then
              let assigns := String.intercalate " " (dsts.enum.map (fun (i, r) =>
                let lane := s!"_gx_v.lane[{i}]"
                writeLoaded r lane))
              "    { auto _gx_v = gx_gld_vec<" ++ mt.cname ++ ", " ++
                toString n ++ ">(" ++ base ++ "); " ++ assigns ++ " }"
            else String.intercalate " " (dsts.enum.map (fun (i, r) =>
              let a := s!"(({base}) + {CMem.elemOff i mt.bytes})"
              s!"    {writeLoaded r (loadE mt.cname a)}"))
      /- PTX definition: `st` writes typed scalar or vector source elements to consecutive addresses in the selected state space. -/
      /- Lean modeling: source arity comes from `.vec`, cache-policy operands are separated from stored values, and unsupported types/arity routes trap; shared::cluster addresses decode through the `trusted:cluster-runtime` helper; address/offset values are shallow-modeled while the C store remains trusted. -/
      | "st"   =>
        let supportedTy := ["b8", "u8", "s8", "b16", "u16", "s16",
                            "b32", "u32", "s32", "b64", "u64", "s64", "f32", "f64"]
        let memTok := toks.getLastD ""
        let vecToks := toks.filter (fun t => t == "v2" || t == "v4" || t == "v8")
        let typeToks := toks.filter (fun t => supportedTy.contains t)
        let sems := ["weak", "volatile", "relaxed", "release"]
        let scopes := ["cta", "cluster", "gpu", "sys"]
        let spaces := ["global", "local", "param", "param::func", "shared", "shared::cta",
                       "shared::cluster"]
        let cacheOps := ["wb", "cg", "cs", "wt"]
        let l1Hints := ["L1::evict_normal", "L1::evict_unchanged", "L1::evict_first",
                        "L1::evict_last", "L1::no_allocate"]
        let l2Hints := ["L2::evict_normal", "L2::evict_first", "L2::evict_last", "L2::cache_hint"]
        let q := (toks.drop 1).take (toks.length - 2)
        let semQ := q.filter (fun t => sems.contains t)
        let scopeQ := q.filter (fun t => scopes.contains t)
        let spaceQ := q.filter (fun t => spaces.contains t)
        let cacheQ := q.filter (fun t => cacheOps.contains t)
        let l1Q := q.filter (fun t => l1Hints.contains t)
        let l2Q := q.filter (fun t => l2Hints.contains t)
        let vecQ := q.filter (fun t => t == "v2" || t == "v4" || t == "v8")
        let semScopeOK :=
          if toks.contains "relaxed" || toks.contains "release" then scopeQ.length == 1
          else scopeQ.isEmpty
        let syntaxOK :=
          q.all (fun t => sems.contains t || scopes.contains t || spaces.contains t ||
            cacheOps.contains t || l1Hints.contains t || l2Hints.contains t ||
            t == "v2" || t == "v4" || t == "v8") &&
          semQ.length ≤ 1 && scopeQ.length ≤ 1 && spaceQ.length ≤ 1 && cacheQ.length ≤ 1 &&
          -- PTX §9.7.9.11 examples likewise put the state space before
          -- sem/scope, so accept documented permutations of unique categories.
          l1Q.length ≤ 1 && l2Q.length ≤ 1 && vecQ.length ≤ 1 && semScopeOK
        let n := if toks.contains "v8" then 8 else if toks.contains "v4" then 4
                 else if toks.contains "v2" then 2 else 1
        let expectedArgs := n + 1 + (if toks.contains "L2::cache_hint" then 1 else 0)
        if memTok == "b128" then
          match args with
          | [a, v] =>
            if isShared && vecToks.isEmpty then
              "    { unsigned __int128 _gx_q = (unsigned __int128)(" ++ oC v ++ "); memcpy((void*)(uintptr_t)(" ++
                mkAddr a ++ "), &_gx_q, 16); }"
            else trapOp
          | _ => trapOp
        else if !syntaxOK || vecToks.length > 1 || typeToks.length != 1 ||
            !supportedTy.contains memTok || args.length != expectedArgs then trapOp
        else
          let mt := memTyOf memTok
          let st1 : String → String → String := fun addr val =>
            if guarded then s!"    gx_gst<{mt.cname}>({addr}, ({mt.cname})({val}));"
            else if deviceAddress then
              s!"    if (!gx_fake({addr})) *({mt.cname} {vol}*)(uintptr_t)({addr}) = {val};"
            else s!"    *({mt.cname} {vol}*)(uintptr_t)({addr}) = {val};"
          let rawAddr := args.headD ""
          let addr := if rawAddr.endsWith ".unified" then rawAddr.dropRight 8 else rawAddr
          let base := mkAddr addr
          let vals := (args.drop 1).take n
          let storeTyped (r : String) : String :=
            let value := oC r
            if memTok == "f32" || memTok == "f64" then floatOperand regs memTok r
            else if memTok == "b32" || memTok == "b64" then bitOperand regs memTok r
            else value
          if deviceAddress && toks.contains "volatile" && n == 4 && memTok == "u32" then
            let vectorVals := String.intercalate ", " (vals.map (fun r =>
              s!"(uint32_t)({storeTyped r})"))
            s!"    gx_gst_v4_u32({base}, {vectorVals});"
          else if n == 1 then st1 s!"({base})" (storeTyped (vals.headD "0"))
          else if guarded then
            let vectorVals := String.intercalate ", " (vals.map (fun r =>
              s!"({mt.cname})({storeTyped r})"))
            "    { GxGlobalVec<" ++ mt.cname ++ ", " ++ toString n ++
              "> _gx_v = {{ " ++ vectorVals ++ " }}; gx_gst_vec(" ++
              base ++ ", _gx_v); }"
          else String.intercalate " " (vals.enum.map (fun (i, r) =>
            st1 s!"(({base}) + {CMem.elemOff i mt.bytes})" (storeTyped r)))
      /- PTX definition: `cvt` converts between declared source/destination types under applicable rounding and saturation modifiers. -/
      /- Lean modeling: plain integer/integer plus same-or-widening float routes, mandatory `.rn` integer-to-f32/f64 and supported float narrowing, and decoded float-to-integer rni/rzi/rmi/rpi routes form a closed allowlist; every other spelling traps before printing. -/
      | "cvt"  =>
        -- cvt.<to>.<from> / cvt.<round>.<to>.<from>.
        -- INT→INT: a shallow-value-backed resize template (CCvt.cvtI_equiv).
        -- The printed `(DT)((ST)(x))` remains `trusted:template-print` and
        -- sign-extends exactly when the SOURCE type is
        -- signed. (`(int64_t)(uint32_t_reg)` — the old emission — zero-
        -- extends: that was the live broadcast.i64 all-zeros bug.)
        -- INT→FLOAT: only mandatory-RN f32/f64 destinations are selected;
        -- the fixed-width source cast is value-modeled and IEEE conversion trusted.
        -- FLOAT→INT: the four integer-rounding modes are selected explicitly,
        -- followed by a trusted C conversion/range boundary.
        -- FLOAT→FLOAT: plain identity/widening and three explicit RN narrowing
        -- routes remain; f16/bf16 encodings use gx_h2f/gx_f2h
        -- (bit-reinterpret then value-convert), NOT a value cast of the raw
        -- pattern.
        -- f32/f64 crossings reinterpret the declared C local's bits, including
        -- arbitrary/bare register names. Host conversions remain trusted.
        let cvtTypes := ["u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64",
                         "f16", "bf16", "f32", "f64"]
        let tys := toks.filter (fun t => cvtTypes.contains t)
        let toTok := tys.getD (tys.length - 2) "u32"
        let fromTok := tys.getLastD "u32"
        let round := toks.filterMap (fun q =>
          match q with
          | "rni" => some "nearbyint" | "rzi" => some "trunc"
          | "rmi" => some "floor" | "rpi" => some "ceil" | _ => none) |>.head?
        let plain := toks == ["cvt", toTok, fromTok]
        let floatRoundNearest := toks == ["cvt", "rn", toTok, fromTok]
        let floatToIntRounded :=
          match toks with
          | ["cvt", r, toTy, fromTy] =>
            ["rni", "rzi", "rmi", "rpi"].contains r && toTy == toTok && fromTy == fromTok
          | _ => false
        let intToFloatRN :=
          floatRoundNearest && (toTok == "f32" || toTok == "f64" || (toTok == "f16" && fromTok == "s32"))
        let plainFloatPair :=
          toTok == fromTok ||
          ((fromTok == "f16" || fromTok == "bf16") && (toTok == "f32" || toTok == "f64")) ||
          (fromTok == "f32" && toTok == "f64")
        let narrowingFloatRN :=
          floatRoundNearest &&
          ((toTok == "f32" && fromTok == "f64") ||
           (toTok == "f16" && fromTok == "f32") ||
           (toTok == "bf16" && fromTok == "f32"))
        let floatSrc (tok x : String) : String :=
          if tok == "f16" then s!"gx_h2f((uint16_t)({oC x}))"
          else if tok == "bf16" then s!"gx_bf2f((uint16_t)({oC x}))"
          else if tok == "f32" || tok == "f64" then floatOperand regs tok x
          else oC x
        let floatDst (tok d value : String) : String :=
          if tok == "f32" || tok == "f64" then writeFloat tok d value
          else s!"    {dest d} = {value};"
        let routeModeled :=
          match cvtIntTy? toTok, cvtIntTy? fromTok with
          | some _, some _ => plain
          | none, some _ => intToFloatRN
          | some _, none => floatToIntRounded
          | none, none => (plain && plainFloatPair) || narrowingFloatRN
        let irndMode : Option String :=
          match toks with
          | "cvt" :: r :: rest =>
            let mode := match r with
              | "rni" => some "0" | "rzi" => some "1" | "rmi" => some "2" | "rpi" => some "3" | _ => none
            let ty := rest.getLastD ""
            let mods := rest.take (rest.length - 2)
            if rest.length >= 2 && rest.getD (rest.length - 2) "" == ty &&
               ((ty == "f32" && [[], ["ftz"], ["sat"], ["ftz", "sat"]].contains mods) ||
                (ty == "f64" && [[], ["sat"]].contains mods)) then mode else none
          | _ => none
        let narrowMode : Option Nat :=
          match toks with
          | "cvt" :: r :: rest =>
            let mode := match r with
              | "rn" => some 0 | "rz" => some 1 | "rm" => some 2 | "rp" => some 3 | _ => none
            let mods := rest.take (rest.length - 2)
            if rest.length >= 2 && rest.drop (rest.length - 2) == ["f32", "f64"] &&
                [[], ["ftz"], ["sat"], ["ftz", "sat"]].contains mods then mode else none
          | _ => none
        let tf32Mode : Option Nat :=
          match toks with
          | "cvt" :: r :: rest =>
            let mode := match r with
              | "rn" => some 0 | "rz" => some 1 | "rna" => some 2 | _ => none
            let mods := rest.take (rest.length - 2)
            let validMods := if r == "rna" then [[], ["satfinite"]]
              else [[], ["satfinite"], ["relu"], ["satfinite", "relu"]]
            if rest.length >= 2 && rest.drop (rest.length - 2) == ["tf32", "f32"] &&
                validMods.contains mods then mode else none
          | _ => none
        if tf32Mode.isSome then
          match args with
          | [d, a] =>
            if d == "_" then trapOp
            else s!"    gx_movb({dest d}, gx_cvt_tf32_f32({tf32Mode.getD 0}u, {if toks.contains "satfinite" then "true" else "false"}, {if toks.contains "relu" then "true" else "false"}, {floatSrc "f32" a}));"
          | _ => trapOp
        else if narrowMode.isSome then
          match args with
          | [d, a] =>
            if d == "_" then trapOp
            else writeFloat "f32" d s!"gx_cvt_f64_f32({narrowMode.getD 0}u, {if toks.contains "ftz" then "true" else "false"}, {if toks.contains "sat" then "true" else "false"}, {floatSrc "f64" a})"
          | _ => trapOp
        else if toks == ["cvt", "sat", "f32", "f32"] then
          match args with
          | [d, a] => if d == "_" then trapOp else writeFloat "f32" d s!"gx_cvt_sat_f32({floatSrc "f32" a})"
          | _ => trapOp
        -- Same-type float integer rounding cvt{.irnd}{.ftz}{.sat}.ftype.ftype
        -- (f32 or f64; .ftz only on f32) keeps floating storage. Any other
        -- modifier, order or operand shape traps.
        else if irndMode.isSome then
          match args with
          | [d, a] =>
            if d == "_" then trapOp
            else
              let mode := irndMode.getD "0"
              let sat := if toks.contains "sat" then "1" else "0"
              if toTok == "f32" then
                let ftz := if toks.contains "ftz" then "1" else "0"
                writeFloat "f32" d s!"gx_cvt_irnd_f32({mode}, {ftz}, {sat}, {floatSrc "f32" a})"
              else writeFloat "f64" d s!"gx_cvt_irnd_f64({mode}, {sat}, {floatSrc "f64" a})"
          | _ => trapOp
        else if (toks.take 2 == ["cvt", "rn"]) && toks.getLast? != none &&
                ["e2m1x2", "e2m3x2", "e3m2x2", "e5m2x2"].contains (toks.getD (toks.length - 1) "") &&
                (toks.drop 2 == ["f16x2", toks.getD (toks.length - 1) ""] ||
                 toks.drop 2 == ["relu", "f16x2", toks.getD (toks.length - 1) ""]) then
          -- Unpack two narrow values to f16x2 (upper lane to the upper half).
          let fmt := match toks.getD (toks.length - 1) "" with
            | "e5m2x2" => 1 | "e2m3x2" => 2 | "e3m2x2" => 3 | _ => 4
          match args with
          | [d, a] =>
            if d == "_" then trapOp
            else s!"    gx_movb({dest d}, gx_cvt_rn_f16x2_minifloat({fmt}, (uint32_t)({oC a}), {if toks.contains "relu" then "true" else "false"}));"
          | _ => trapOp
        else if toks.take 3 == ["cvt", "rn", "satfinite"] && toks.getLast? == some "f32" &&
                (toks.length == 5 || (toks.length == 6 && toks.getD 3 "" == "relu")) &&
                ["e2m1x2", "e2m3x2", "e3m2x2", "e5m2x2"].contains (toks.getD (toks.length - 2) "") then
          -- Pack two f32 values into narrow floats (first source to the upper lane).
          let fmt := match toks.getD (toks.length - 2) "" with
            | "e5m2x2" => 1 | "e2m3x2" => 2 | "e3m2x2" => 3 | _ => 4
          match args with
          | [d, a, b] =>
            if d == "_" then trapOp
            else s!"    gx_movb({dest d}, ({if fmt == 4 then "uint8_t" else "uint16_t"})gx_cvt_pack_minifloat_f32({fmt}, {floatSrc "f32" a}, {floatSrc "f32" b}, {if toks.contains "relu" then "true" else "false"}));"
          | _ => trapOp
        else if (toks == ["cvt", "rz", "satfinite", "ue8m0x2", "f32"] || toks == ["cvt", "rp", "satfinite", "ue8m0x2", "f32"] ||
                 toks == ["cvt", "rz", "ue8m0x2", "f32"] || toks == ["cvt", "rp", "ue8m0x2", "f32"]) then
          match args with
          | [d, a, b] =>
            if d == "_" then trapOp
            else s!"    gx_movb({dest d}, gx_cvt_ue8m0x2_f32({floatSrc "f32" a}, {floatSrc "f32" b}, {if toks.contains "rp" then "true" else "false"}));"
          | _ => trapOp
        else if (toks == ["cvt", "rz", "satfinite", "ue8m0x2", "bf16x2"] || toks == ["cvt", "rp", "satfinite", "ue8m0x2", "bf16x2"] ||
                 toks == ["cvt", "rz", "ue8m0x2", "bf16x2"] || toks == ["cvt", "rp", "ue8m0x2", "bf16x2"]) then
          match args with
          | [d, a] =>
            if d == "_" then trapOp
            else s!"    gx_movb({dest d}, gx_cvt_ue8m0x2_bf16x2((uint32_t)({oC a}), {if toks.contains "rp" then "true" else "false"}));"
          | _ => trapOp
        else if toks == ["cvt", "rn", "bf16x2", "ue8m0x2"] then
          match args with
          | [d, a] =>
            if d == "_" then trapOp
            else s!"    gx_movb({dest d}, gx_cvt_rn_bf16x2_ue8m0x2((uint32_t)({oC a})));"
          | _ => trapOp
        else if toks == ["cvt", "rn", "f16x2", "e4m3x2"] then
          match args with
          | [d, a] =>
            if d == "_" then trapOp
            else s!"    gx_movb({dest d}, gx_cvt_rn_f16x2_e4m3x2((uint16_t)({oC a})));"
          | _ => trapOp
        else if toks == ["cvt", "rn", "satfinite", "e4m3x2", "f32"] then
          match args with
          | [d, a, b] =>
            if d == "_" then trapOp
            else s!"    gx_movb({dest d}, gx_cvt_rn_satfinite_e4m3x2_f32({floatSrc "f32" a}, {floatSrc "f32" b}));"
          | _ => trapOp
        else if toks == ["cvt", "rn", "bf16x2", "f32"] then
          match args with
          | [d, a, b] =>
            if d == "_" then trapOp
            else s!"    {dest d} = gx_cvt_rn_bf16x2_f32({floatSrc "f32" a}, {floatSrc "f32" b});"
          | _ => trapOp
        else if tys.length != 2 || !routeModeled then trapOp
        else match args with
        | [d, a] =>
          if d == "_" then trapOp
          else
          match cvtIntTy? toTok, cvtIntTy? fromTok with
          | some (ds, dw), some (ss, sw) =>
            s!"    {dest d} = {printCvtII ss sw ds dw (oC a)};"
          | none, some (ss, sw) =>
            -- int → float allowlist has only mandatory-RN f32/f64 destinations.
            let value := if toTok == "f16" then s!"gx_f2h((float)(int32_t)({oC a}))"
                         else printCvtIF ss sw (ctypeOf toTok) (oC a)
            floatDst toTok d value
          | some _, none =>
            -- float → int: rounding mode then C conversion (trusted)
            let decoded := floatSrc fromTok a
            let src := match round with
              | some fn => s!"{fn}((double){decoded})"
              | none    => decoded
            s!"    {dest d} = ({ctypeOf toTok})({src});"
          | none, none =>
            -- Same-format half/bfloat conversions preserve the raw register
            -- pattern. Narrowing from f32 uses one native conversion; widening
            -- first decodes the 16-bit format exactly. f64→half/bfloat and
            -- f16↔bf16 are rejected above to avoid double rounding or treating
            -- a raw 16-bit encoding as an integer value.
            if toTok == fromTok && (toTok == "f16" || toTok == "bf16") then
              s!"    {dest d} = {oC a};"
            else
              let src := floatSrc fromTok a
              let value :=
                if toTok == "f16" then s!"gx_f2h((float)({src}))"
                else if toTok == "bf16" then s!"gx_f2bf((float)({src}))"
                else s!"({ctypeOf toTok})({src})"
              floatDst toTok d value
        | _ => trapOp
      -- ==================================================================
      -- BIT MANIPULATION: COUNT AND REVERSE
      -- ==================================================================
      /- PTX definition: `popc.b32/.b64` returns the number of one bits in the source value. -/
      /- Lean modeling: helper width selection and operand order preserve production behavior, but helper/compiler correctness remains the explicit `trusted:popc-clz` boundary. -/
      | "popc" => match args with
                  | d :: a :: _ => s!"    {dest d} = {printPopc (tyTok.endsWith "64") (oC a)};"
                  | _ => s!"    /* malformed: {s} */"
      /- PTX definition: `clz.b32/.b64` returns the number of leading zero bits in the source value. -/
      /- Lean modeling: helper width selection and operand order preserve production behavior, but helper/compiler correctness remains the explicit `trusted:popc-clz` boundary. -/
      | "clz"  => match args with
                  | d :: a :: _ => s!"    {dest d} = {printClz (tyTok.endsWith "64") (oC a)};"
                  | _ => s!"    /* malformed: {s} */"
      /- PTX definition: `brev.b32/.b64` reverses the bit order of the source value. -/
      /- Lean modeling: helper width selection and operand order preserve production behavior, but helper/compiler correctness remains the explicit `trusted:brev` boundary. -/
      | "brev" => match args with
                  | d :: a :: _ => s!"    {dest d} = {printBrev (tyTok.endsWith "64") (oC a)};"
                  | _ => s!"    /* malformed: {s} */"
      -- ==================================================================
      -- COMMUNICATION AND CONTROL
      -- ==================================================================
      /- PTX definition: direct/indirect calls pass ordered input parameters and optional return parameters through `.param` objects. -/
      /- Lean modeling: generated pointer arrays preserve tuple order and unknown/malformed return calls trap; ABI, control-flow, and adapter behavior remain `structural:sreg-ctrl` trusted plumbing. -/
      | "call" =>
        -- Direct/indirect input calls:
        --   call.uni fname, (param0, param1);
        --   call %rd, (param0), prototype_N;
        -- Return-valued internal calls still require translated `.func` return
        -- signatures. CUDA external functions use GX's explicit adapter ABI.
        let hasResult := args.head?.any (fun a => (strim a).startsWith "(")
        let outputs := if hasResult then callInputParams (args.headD "") else []
        let callArgs := if hasResult then args.drop 1 else args
        match callArgs with
        | callee :: tail =>
          let c := strim (callee.replace "," "")
          let inputs := match tail.find? (fun a => (strim a).startsWith "(") with
            | some group => callInputParams group
            | none => []
          if c == "__drperf_gpu_device_begin_v1" then
            "    gx_ptx_trap_op(\"legacy device PCVs: regenerate with entry-PCV constructor\");"
          else if c == "__drperf_gpu_device_ready_v2" then
            if outputs.isEmpty && inputs.isEmpty then "    drperf_gpu_device_ready();"
            else "    gx_ptx_trap_op(\"drperf_gpu_device_ready.abi\");"
          else if c == "__drperf_gpu_device_begin_v2" then
            match outputs, inputs with
            | [], [name, scope] =>
                s!"    drperf_gpu_device_begin((const char*)(uintptr_t)(*(const uint64_t*)(uintptr_t)({oC name})), *(const uint32_t*)(uintptr_t)({oC scope}));"
            | _, _ => "    gx_ptx_trap_op(\"drperf_gpu_device_begin.abi\");"
          else if c == "__drperf_gpu_device_pcv_v1" then
            match outputs, inputs with
            | [], [name, value] =>
                s!"    drperf_gpu_device_pcv((const char*)(uintptr_t)(*(const uint64_t*)(uintptr_t)({oC name})), *(const int64_t*)(uintptr_t)({oC value}));"
            | _, _ => "    gx_ptx_trap_op(\"drperf_gpu_device_pcv.abi\");"
          else if c == "__drperf_gpu_device_end_v1" then
            if outputs.isEmpty && inputs.isEmpty then "    drperf_gpu_device_end();"
            else "    gx_ptx_trap_op(\"drperf_gpu_device_end.abi\");"
          else if (c.splitOn "gx_ptx_uniform_load64").length > 1 then
            emitUniformLoad64Call outputs inputs
          else if (c.splitOn "gx_ptx_counted_load32").length > 1 then
            emitCountedLoad32Call outputs inputs
          else if (c.splitOn "gx_ptx_counted_load8").length > 1 then
            emitCountedLoad8Call outputs inputs
          else if (c.splitOn "gx_ptx_cohort_barrier").length > 1 then
            emitCohortBarrierCall outputs inputs
          else if (c.splitOn "gx_ptx_block_cohort_width").length > 1 then
            emitBlockCohortWidthCall outputs inputs
          else if (c.splitOn "gx_ptx_warp_cohort_width").length > 1 then
            emitWarpCohortWidthCall outputs inputs
          else if (c.splitOn "gx_ptx_independent_block_cohort_width").length > 1 then
            emitBlockCohortWidthCall outputs inputs
          else if (c.splitOn "gx_ptx_masked_column_scan_u32").length > 1 then
            emitMaskedColumnScanCall outputs inputs
          else if (c.splitOn "gx_ptx_block_transfer_u32").length > 1 then
            emitBlockTransferU32Call outputs inputs
          else if (c.splitOn "gx_ptx_masked_column_add_u32").length > 1 then
            emitMaskedColumnAddCall outputs inputs
          else if (c.splitOn "gx_ptx_counted_copy").length > 1 then
            emitCountedCopyCall outputs inputs
          else if (c.splitOn "gx_ptx_ballot_columns").length > 1 then
            emitWarpBallotColumnsCall outputs inputs
          else if (c.splitOn "gx_ptx_mbarrier_wait").length > 1 then
            emitMbarrierWaitCall outputs inputs
          else if (c.splitOn "gx_ptx_shared_index_gather_f32").length > 1 then
            emitSharedIndexGatherCall outputs inputs
          else if (c.splitOn "gx_ptx_block_copy16").length > 1 then
            if outputs.isEmpty then emitBlockCopy16Call inputs
            else "    gx_ptx_trap_op(\"gx_ptx_block_copy16.abi\");"
          else if (c.splitOn "gx_ptx_block_scan_u32").length > 1 then
            emitBlockScanCall outputs inputs
          else if (c.splitOn "gx_ptx_grid_sync").length > 1 then
            if inputs.isEmpty && outputs.isEmpty then "    gx_grid_sync();"
            else "    gx_ptx_trap_op(\"gx_ptx_grid_sync.abi\");"
          else if (c.splitOn "gx_ptx_nccl_abort_poll").length > 1 then
            match inputs with
            | [address] =>
                s!"    gx_nccl_abort_poll(*(const uint64_t*)(uintptr_t)({oC address}));"
            | _ => "    gx_ptx_trap_op(\"gx_ptx_nccl_abort_poll.abi\");"
          else if (c.splitOn "gx_ptx_payload_event").length > 1 ||
                  (c.splitOn "gx_ptx_memory_event").length > 1 then
            emitPayloadEventCall inputs
          else if (c.splitOn "gx_ptx_payload_scope").length > 1 then
            emitPayloadScopeCall inputs
          else if (c.splitOn "gx_ptx_payload_needed").length > 1 then
            emitPayloadNeededCall outputs inputs
          else if (c.splitOn "gx_ptx_control_load_16").length > 1 then
            match outputs, inputs with
            | [], [dst, src] =>
                s!"    gx_ptx_control_load_16(*(const uint64_t*)(uintptr_t)({oC dst}), *(const uint64_t*)(uintptr_t)({oC src}));"
            | _, _ => "    gx_ptx_trap_op(\"gx_ptx_control_load_16.abi\");"
          else if (c.splitOn "gx_ptx_control_poll").length > 1 then
            if inputs.isEmpty && outputs.isEmpty then "    gx_cpu_kernel_poll();"
            else "    gx_ptx_trap_op(\"gx_ptx_control_poll.abi\");"
          else if (c.splitOn "gx_ptx_control_load_acquire_sys_u32").length > 1 then
            (emitPollLoadCall false outputs inputs).replace "gx_ptx_poll_load<uint32_t>" "gx_ptx_poll_load<uint32_t, __ATOMIC_ACQUIRE, false>"
          else if (c.splitOn "gx_ptx_control_load_sys_u32").length > 1 then
            (emitPollLoadCall false outputs inputs).replace "gx_ptx_poll_load<uint32_t>" "gx_ptx_poll_load<uint32_t, __ATOMIC_RELAXED, false>"
          else if (c.splitOn "gx_ptx_poll_sys_u32").length > 1 then
            emitPollLoadCall false outputs inputs
          else if (c.splitOn "gx_ptx_poll_sys_u64").length > 1 ||
                  (c.splitOn "gx_ptx_poll_gpu_u64").length > 1 then
            emitPollLoadCall true outputs inputs
          else
            match externalCallTarget c with
            | some target => emitExternalCall target outputs inputs
            | none =>
              if hasResult then
                "    gx_ptx_trap_op(\"call.return\");"
              else
                emitInputCall c (c.startsWith "%") inputs
        | _ => "    gx_ptx_trap_op(\"call.malformed\");"
      /- PTX definition: `fence`/`membar` order this thread's earlier and later memory operations at the declared scope or legacy level. -/
      /- Lean modeling: exact `.cta` scope emits a compiler barrier and every wider/restricted/proxy form emits a full host fence; PTX proxy, cluster, and formal memory-model equivalence remain explicit trusted/gap boundaries. -/
      | "fence" | "membar" => "    " ++ printMemoryFence (toks.contains "cta")
      /- PTX definition: `griddepcontrol` instructions manage dependent-grid launch ordering. -/
      /- Lean modeling: the no-op retains CPU execution plumbing only and remains explicitly qualified by `gap:sync-runtime-limits`, with no PTX ordering proof. -/
      | "griddepcontrol" => ";"
      /- PTX definition: `bar` synchronizes participating threads and `bar.warp.sync` carries a member mask. -/
      /- Lean modeling: helper selection preserves the runtime route but synchronization semantics are structural and explicitly limited by `gap:sync-runtime-limits`. -/

      /- PTX definition: `cvta` converts addresses between named state spaces and the generic address space. -/
      /- Lean modeling: addition/subtraction against `_gx_shm` matches GX's shared-window representation; that representation is the explicit `trusted:shared-window` boundary. -/
      | "cvta" => match args with
                  | d :: a :: _ =>
                    let src := oC a
                    if isShared && toks.contains "to" then
                      s!"    {dest d} = (uint64_t)((char*)(uintptr_t){src} - _gx_shm) | gx_shared_rank_bits();"
                    else if isShared then
                      s!"    {dest d} = {sharedPtr src};"
                    else s!"    {dest d} = {src};"
                  | _ => s!"    /* malformed: {s} */"
      /- PTX definition: `vote`/`vote.sync` combine active-lane predicates, with sync forms using a member mask. -/
      /- Lean modeling: exact sync any/all/ballot forms preserve mode, mask, and optional predicate negation; uni, deprecated non-sync, sink, and malformed forms trap while warp rendezvous remains structural runtime trust. -/
      | "vote" =>
        let predC (p : String) : Option String :=
          let neg := p.startsWith "!"
          let src := if neg then p.drop 1 else p
          if src.isEmpty || src.startsWith "!" then none
          else
            let v := oC src
            some (if neg then s!"(!{v})" else v)
        match toks, args with
        | ["vote", "sync", "any", "pred"], [d, p, m] =>
          if d == "_" then trapOp else match predC p with
          | some v => s!"    {dest d} = __any_sync((unsigned){oC m}, (int){v});"
          | none => trapOp
        | ["vote", "sync", "all", "pred"], [d, p, m] =>
          if d == "_" then trapOp else match predC p with
          | some v => s!"    {dest d} = __all_sync((unsigned){oC m}, (int){v});"
          | none => trapOp
        | ["vote", "sync", "ballot", "b32"], [d, p, m] =>
          if d == "_" then trapOp else match predC p with
          | some v => s!"    {dest d} = __ballot_sync((unsigned){oC m}, (int){v});"
          | none => trapOp
        -- `.uni`, deprecated non-sync vote, and malformed/mis-typed forms
        -- have no faithful runtime route yet.
        | _, _ => trapOp
      /- PTX definition: redux.sync reduces integer values over the member mask, returning one result to every participant. -/
      /- Lean modeling: signed/unsigned add/min/max and bitwise b32 forms use a trusted warp rendezvous; other forms trap. -/
      | "redux" =>
        match toks, args with
        | ["redux", "sync", reduceOp, ty], [d, value, mask] =>
          let kind := match reduceOp, ty with
            | "add", "u32" | "add", "s32" => 0
            | "min", "u32" => 1
            | "max", "u32" => 2
            | "min", "s32" => 3
            | "max", "s32" => 4
            | "and", "b32" => 5
            | "or", "b32" => 6
            | "xor", "b32" => 7
            | _, _ => 8
          if d == "_" || kind == 8 then trapOp
          else s!"    gx_movb({dest d}, gx_redux_sync((uint32_t)({bitOperand regs "b32" value}), (uint32_t)({oC mask}), {kind}));"
        | _, _ => trapOp
      /- PTX definition: `shfl.sync.idx/bfly.b32` exchanges a lane value and reports whether the selected source lane participates. -/
      /- Lean modeling: lane-selection values have shallow equality and operand order is preserved; printed/runtime warp exchange is trusted and unsupported forms remain `gap:shfl-forms`. -/
      | "shfl" =>
        -- PTX shfl.sync.{idx,bfly}.b32 has a paired `value|predicate`
        -- destination. gx_shfl_sync owns the warp rendezvous/data exchange;
        -- the lane-selection arithmetic is shared by both supported modes.
        match toks, args with
        | ["shfl", "sync", mode, "b32"], [dp, a, b, c, mask] =>
          if ! (mode == "idx" || mode == "bfly" || mode == "down" || mode == "up") then trapOp
          else match dp.splitOn "|" with
          | [d, p] =>
            if d.isEmpty || p.isEmpty || d == "_" || p == "_" then trapOp
            else
              let tpl : CommTemplate := if mode == "down" then .shflDown
                else if mode == "up" then .shflUp
                else if mode == "bfly" then .shflBfly else .shflIdx
              "    " ++ tpl.print (dest d) (dest p) (oC a) (oC b) (oC c) (oC mask) ""
          | [d] =>
            if d.isEmpty || d == "_" then trapOp else
              let tpl : CommTemplate := if mode == "down" then .shflDown
                else if mode == "up" then .shflUp
                else if mode == "bfly" then .shflBfly else .shflIdx
              "    { int _gx_ignored_pred; " ++
                tpl.print (dest d) "_gx_ignored_pred" (oC a) (oC b) (oC c) (oC mask) "" ++ " }"
          | _ => trapOp
        | _, _ => trapOp
      /- PTX definition: `elect.sync` chooses one active lane from the member mask and reports election status. -/
      /- Lean modeling: election values have shallow equality and destination order is preserved; helper printing and warp execution remain structural trusted boundaries. -/
      | "elect" =>
        match toks, args with
        | ["elect", "sync"], [dp, mask] =>
          match dp.splitOn "|" with
          | [d, p] =>
            if p == "_" then trapOp
            else if d == "_" then s!"    {dest p} = gx_elect_pred((uint32_t){oC mask});"
            else "    " ++ CommTemplate.elect.print (dest d) (dest p) (oC mask) "" "" "" ""
          | _ => trapOp
        | _, _ => trapOp
      /- PTX definition: `red.add` atomically adds to memory without returning its old value. -/
      /- Lean modeling: the narrow global integer route reuses the real cross-process atomic helper, discarding its result; other forms trap. -/
      | "red" =>
        let ty := toks.getLastD ""
        let q := (toks.drop 1).take (toks.length - 3)
        let semQ := q.filter (fun t => t == "relaxed" || t == "release")
        let scopeQ := q.filter (fun t => ["cta", "cluster", "gpu", "sys"].contains t)
        let spaceQ := q.filter (fun t => t == "global")
        if toks.getD (toks.length - 2) "" != "add" ||
            !["u32", "s32", "u64"].contains ty ||
            !q.all (fun t => ["relaxed", "release", "cta", "cluster", "gpu", "sys", "global"].contains t) ||
            semQ.length > 1 || scopeQ.length > 1 || spaceQ.length != 1 then (atomRoute false).getD trapOp
        else match args with
        | [m, value] =>
          s!"    (void)gx_atom_add<{ctypeOf ty}>({mkAddr m}, ({ctypeOf ty})({oC value}));"
        | _ => trapOp
      /- PTX definition: `atom` performs an indivisible read-modify-write selected by operation, type, state space, scope, and ordering qualifiers. -/
      /- Lean modeling: an ISA-derived scalar operation/type allowlist selects integral host-atomic helpers; the remaining scalar Table 35 forms (bit ops, inc/dec, f32/f64/f16/bf16(x2) add, b128 exch/cas, red without destination) go through the CAS-based `gx_atom_rmw`/`gx_atom_b128` helpers; vector forms and mismatched type/op pairs trap, shared::cluster addresses decode through the `trusted:cluster-runtime` helper, while atomic runtime/order remains trusted. -/
      | "atom" =>
        let addr (m : String) := mkAddr m
        let atomTy := toks.getLastD ""
        let atomCt := ctypeOf atomTy
        let atomOps := ["and", "or", "xor", "add", "inc", "dec", "min", "max", "cas", "exch"]
        let atomTypes := ["b16", "b32", "b64", "b128", "u32", "u64", "s32", "s64",
                          "f16", "f16x2", "bf16", "bf16x2", "f32", "f64"]
        -- nvcc emits both qualifier-first `atom.global.add.u32` and
        -- operation-first `atom.add.release.gpu.u32` spellings. Normalize the
        -- latter before applying the same closed qualifier/type allowlist.
        let operationFirst := atomOps.contains (toks.getD 1 "")
        let atomOp := if operationFirst then toks.getD 1 ""
                      else toks.getD (toks.length - 2) ""
        let q := if operationFirst then (toks.drop 2).take (toks.length - 3)
                 else (toks.drop 1).take (toks.length - 3)
        let sems := ["relaxed", "acquire", "release", "acq_rel"]
        let scopes := ["cta", "cluster", "gpu", "sys"]
        let spaces := ["global", "shared", "shared::cta", "shared::cluster"]
        let semQ := q.filter (fun t => sems.contains t)
        let scopeQ := q.filter (fun t => scopes.contains t)
        let spaceQ := q.filter (fun t => spaces.contains t)
        let syntaxOK := atomOps.contains atomOp && atomTypes.contains atomTy &&
          q.all (fun t => sems.contains t || scopes.contains t || spaces.contains t) &&
          -- PTX §9.7.14.5 and nvcc output place space/scope/sem in more than
          -- one order, so require unique recognized qualifiers without
          -- inventing a canonical order.
          semQ.length ≤ 1 && scopeQ.length ≤ 1 && spaceQ.length ≤ 1
        if !syntaxOK then (atomRoute true).getD trapOp
        else if atomOp == "cas" && ["b16", "b32", "b64"].contains atomTy then
          match args with
          | d :: m :: cmp :: value :: _ =>
            if d != "_" && args.length == 4 then
              "    " ++ CommTemplate.atomCas.print (dest d) "" (addr m) (oC cmp) (oC value) "" atomCt
            else trapOp
          | _ => trapOp
        else if atomOp == "exch" && ["b32", "b64"].contains atomTy then
          match args with
          | d :: m :: value :: _ =>
            if d != "_" && args.length == 3 then
              "    " ++ CommTemplate.atomExch.print (dest d) "" (addr m) (oC value) "" "" atomCt
            else trapOp
          | _ => trapOp
        else if atomOp == "add" && ["u32", "s32", "u64"].contains atomTy then
          match args with
          | d :: m :: value :: _ =>
            if d != "_" && args.length == 3 then
              "    " ++ CommTemplate.atomAdd.print (dest d) "" (addr m) (oC value) "" "" atomCt
            else trapOp
          | _ => trapOp
        else if atomOp == "min" && ["u32", "s32", "u64", "s64"].contains atomTy then
          match args with
          | d :: m :: value :: _ =>
            if d != "_" && args.length == 3 then
              let tpl : CommTemplate := if atomTy.startsWith "s" then .atomMinS else .atomMinU
              "    " ++ tpl.print (dest d) "" (addr m) (oC value) "" "" atomCt
            else trapOp
          | _ => trapOp
        else if atomOp == "max" && ["u32", "s32", "u64", "s64"].contains atomTy then
          match args with
          | d :: m :: value :: _ =>
            if d != "_" && args.length == 3 then
              let tpl : CommTemplate := if atomTy.startsWith "s" then .atomMaxS else .atomMaxU
              "    " ++ tpl.print (dest d) "" (addr m) (oC value) "" "" atomCt
            else trapOp
          | _ => trapOp
        else (atomRoute true).getD trapOp
      /- PTX definition: asynchronous copy families move bytes between designated state spaces and coordinate completion groups. -/
      /- Lean modeling: the byte-copy route assumes an ordinary unmasked legal-operand form after completion; exact L2 cache-policy operands are erased because PTX defines them as performance-only hints, while scheduling, masks/out-of-bounds modes, and helper execution remain structural/trusted with `gap:cp-async-semantics`. -/
      | "cp" =>
        -- Shared helpers for tensor (TMA) and plain bulk forms. Tensor copies
        -- and reductions execute at issue in the runtime; see
        -- emu/ptx/tensor_map.cc and emu/ptx/bulk_copy.cc.
        let ops := (splitOperands rest).map (fun o => strim (if o.endsWith ";" then o.dropRight 1 else o))
        let semScope := ["weak", "relaxed", "cta", "cluster", "gpu", "sys", "b128"]
        let redOpCode (t : String) : Option Nat := match t with
          | "add" => some 0 | "min" => some 1 | "max" => some 2 | "inc" => some 3 | "dec" => some 4
          | "and" => some 5 | "or" => some 6 | "xor" => some 7 | _ => none
        let tensorRequest (group : String) (want : Nat) : Option String :=
          match bracketElements group with
          | tmap :: coords =>
            if coords.length == want then
              let elems := String.intercalate ", " (coords.map (fun c => s!"(int32_t)({oC c})"))
              some ("(uint64_t)(" ++ oC tmap ++ "), std::array<int32_t, " ++ toString want ++ ">{" ++ elems ++ "}.data(), " ++ toString want ++ "u")
            else none
          | _ => none
        if toks.take 4 == ["cp", "async", "bulk", "tensor"] || toks.take 5 == ["cp", "reduce", "async", "bulk", "tensor"] then
          let isReduce := toks.getD 1 "" == "reduce"
          let dimTok := toks.getD (if isReduce then 5 else 4) ""
          let dims := (dimTok.dropRight 1).toNat?.getD 0
          -- `.cta_group::N` may precede the state spaces (CUTLASS's spelling).
          let quals := (toks.drop (if isReduce then 6 else 5)).filter (fun t => !t.startsWith "cta_group::")
          let ctaGroups := toks.filter (fun t => t.startsWith "cta_group::")
          let modeOf (t : String) : Option Nat := match t with
            | "tile" => some 0 | "im2col" => some 1 | "im2col::w" => some 2 | "im2col::w::128" => some 3
            | "tile::gather4" | "tile::scatter4" => some 4 | "im2col_no_offs" => some 5 | _ => none
          let modes := quals.filterMap modeOf
          let mode := modes.headD 0
          let multicast := quals.contains "multicast::cluster"
          let hint := if quals.contains "L2::cache_hint" then 1 else 0
          let groupOK := ctaGroups.length ≤ 1 && ctaGroups.all (fun t => t == "cta_group::1" || t == "cta_group::2")
          let cg := if ctaGroups.contains "cta_group::2" then "2" else "1"
          let ncoords := if mode == 4 then 5 else dims
          let known (extra : List String) := quals.all (fun t => (modeOf t).isSome || ["global", "shared::cluster", "shared::cta",
            "mbarrier::complete_tx::bytes", "bulk_group", "multicast::cluster", "L2::cache_hint"].contains t || extra.contains t)
          -- im2col (convolution) modes are not modelled and trap.
          if dims < 1 || dims > 5 || dimTok.back != 'd' || !groupOK || modes.length > 1 ||
             (mode == 4 && dims != 2) || !(mode == 0 || mode == 4) then trapOp
          else if isReduce then
            -- cp.reduce.async.bulk.tensor.Nd.global.shared::cta.<op>[.tile|.im2col_no_offs].bulk_group[.L2::cache_hint]
            let rops := quals.filterMap redOpCode
            let rmode := mode
            match rops, ops with
            | [rop], group :: src :: extra =>
              if !known ["add", "min", "max", "inc", "dec", "and", "or", "xor"] || !quals.contains "bulk_group" ||
                 quals.take 2 != ["global", "shared::cta"] || !ctaGroups.isEmpty || extra.length != hint ||
                 rmode != 0 then trapOp
              else match tensorRequest group ncoords with
                | some tc => s!"    gx_tma_store_mode({tc}, (uint32_t)({parseAddr src regs}), {rmode}u, {rop});"
                | none => trapOp
            | _, _ => trapOp
          else if quals.take 2 == ["global", "shared::cta"] then
            -- store: .tile | .tile::scatter4 | .im2col_no_offs, bulk_group
            match ops with
            | group :: src :: extra =>
              if !known [] || !quals.contains "bulk_group" || !ctaGroups.isEmpty || extra.length != hint ||
                 multicast then trapOp
              else match tensorRequest group ncoords with
                | some tc => s!"    gx_tma_store_mode({tc}, (uint32_t)({parseAddr src regs}), {mode}u, -1);"
                | none => trapOp
            | _ => trapOp
          else if (quals.headD "" == "shared::cluster" || quals.headD "" == "shared::cta") && quals.getD 1 "" == "global" &&
                  quals.contains "mbarrier::complete_tx::bytes" && known [] && mode != 5 then
            -- load: [dst], [tmap, {coords}], [mbar] [, ctaMask] [, policy]
            match ops with
            | dst :: group :: mbar :: extra =>
              let mask := if multicast then extra.headD "" else "0"
              if extra.length != (if multicast then 1 else 0) + hint || mask.isEmpty then trapOp
              else match tensorRequest group ncoords with
                | some tc =>
                  s!"    gx_tma_load_mode((uint32_t)({parseAddr dst regs}), {tc}, (uint32_t)({parseAddr mbar regs}), " ++
                    s!"{if multicast then "1" else "0"}u, (uint32_t)({oC mask}), {cg}u, {mode}u, nullptr);"
                | none => trapOp
            | _ => trapOp
          else trapOp
        else if toks.take 5 == ["cp", "async", "bulk", "prefetch", "tensor"] ||
                toks.take 5 == ["cp", "async", "bulk", "prefetch", "L2"] then
          -- L2 prefetches have no functional effect.
          ";"
        else if toks.take 4 == ["cp", "reduce", "async", "bulk"] then
          -- cp.reduce.async.bulk[.sem.scope].dst.src.<completion>[.L2::cache_hint].<op>[.noftz].<type>
          let quals := (toks.drop 4).filter (fun t => !semScope.contains t)
          let rops := quals.filterMap redOpCode
          let ty := quals.getLastD ""
          let tyCode : Option Nat := match ty with
            | "u32" | "b32" => some 0 | "s32" => some 1 | "u64" | "b64" => some 2 | "s64" => some 3
            | "f32" => some 5 | "f64" => some 6 | "f16" => some 7 | "bf16" => some 8 | _ => none
          let toCluster := quals.take 2 == ["shared::cluster", "shared::cta"]
          let toGlobal := quals.take 2 == ["global", "shared::cta"]
          let halfOK := !(ty == "f16" || ty == "bf16") || rops != [0] || quals.contains "noftz"
          let allKnown := quals.all (fun t => ["shared::cluster", "shared::cta", "global", "mbarrier::complete_tx::bytes",
            "bulk_group", "L2::cache_hint", "noftz", ty].contains t || (redOpCode t).isSome)
          match rops, tyCode with
          | [rop], some tc =>
            if !allKnown || !halfOK then trapOp
            else if toCluster && quals.contains "mbarrier::complete_tx::bytes" then
              match ops with
              | [d, src, n, mbar] =>
                s!"    gx_bulk_reduce((uint64_t)({parseAddr d regs}), (uint32_t)({parseAddr src regs}), (uint32_t)({oC n}), 1u, (uint32_t)({parseAddr mbar regs}), {rop}u, {tc}u);"
              | _ => trapOp
            else if toGlobal && quals.contains "bulk_group" then
              let hint := if quals.contains "L2::cache_hint" then 1 else 0
              match ops with
              | d :: src :: n :: extra =>
                if extra.length != hint then trapOp
                else s!"    gx_bulk_reduce((uint64_t)({parseAddr d regs}), (uint32_t)({parseAddr src regs}), (uint32_t)({oC n}), 0u, 0u, {rop}u, {tc}u);"
              | _ => trapOp
            else trapOp
          | _, _ => trapOp
        else if ["cp.async.mbarrier.arrive.b64", "cp.async.mbarrier.arrive.shared.b64",
            "cp.async.mbarrier.arrive.shared::cta.b64", "cp.async.mbarrier.arrive.noinc.b64",
            "cp.async.mbarrier.arrive.noinc.shared.b64", "cp.async.mbarrier.arrive.noinc.shared::cta.b64"].contains op then
          match args with
          | [a] =>
            -- Prior copies have completed synchronously. Default arrival has
            -- net zero pending-count change; noinc consumes one arrival.
            if toks.contains "noinc" then
              s!"    (void)gx_mbarrier_arrive({mkAddr a}, 1u, 0u);"
            else ";"
          | _ => trapOp
        else if (toks == ["cp", "async", "commit_group"] ||
            toks == ["cp", "async", "bulk", "commit_group"]) && args.isEmpty then ";"
        else if toks == ["cp", "async", "wait_all"] && args.isEmpty then ";"
        else if (toks == ["cp", "async", "wait_group"] ||
                 toks == ["cp", "async", "bulk", "wait_group"] ||
                 toks == ["cp", "async", "bulk", "wait_group", "read"]) && args.length == 1 then ";"
        else if toks.take 2 == ["cp", "async"] && (toks.getD 2 "" == "ca" || toks.getD 2 "" == "cg") then
          -- cp.async.{ca,cg}.shared{::cta}.global{.L2::cache_hint}{.L2::{64,128,256}B}
          --   [dst], [src], cp-size{, src-size | ignore-src}{, cache-policy}
          -- completes at issue (cp.async.wait_group/wait_all are then no-ops);
          -- bytes past src-size, or all bytes when ignore-src is true, are zero.
          let mods := toks.drop 5
          let hint := mods.headD "" == "L2::cache_hint"
          let prefetch := if hint then mods.drop 1 else mods
          let modsOK := prefetch.length ≤ 1 && prefetch.all (fun t => ["L2::64B", "L2::128B", "L2::256B"].contains t)
          let spaceOK := (toks.getD 3 "" == "shared" || toks.getD 3 "" == "shared::cta") && toks.getD 4 "" == "global"
          let args := (args.map (fun a => strim (if a.endsWith ";" then a.dropRight 1 else a))).filter (! ·.isEmpty)
          match args with
          | d :: a :: n :: more =>
            let extra := if hint then more.dropLast else more
            let sizeOK := ["4", "8", "16"].contains n && (toks.getD 2 "" != "cg" || n == "16")
            if !modsOK || !spaceOK || !sizeOK || (hint && more.isEmpty) || extra.length > 1 then trapOp
            else
              let dst := sharedPtr (parseAddr d regs)
              match extra with
              | [] => "    " ++ CommTemplate.copyBytes.print "" "" dst (parseAddr a) (oC n) "1" ""
              | [x] =>
                match x.toNat? with
                | some sourceBytes =>
                  -- immediate src-size: a value above cp-size is rejected statically
                  if sourceBytes > n.toNat! then trapOp else
                    s!"    gx_ptx_memcpy({dst}, {parseAddr a}, {sourceBytes}u, 1);" ++
                      s!" memset((char*)(uintptr_t)({dst}) + {sourceBytes}u, 0, {n.toNat! - sourceBytes}u);"
                | none =>
                -- predicate operand: ignore-src; integer register: src-size
                let bytes := if regs.contains (x ++ "#pred") then s!"(({oC x}) ? 0u : {n}u)"
                  else s!"(uint32_t)({oC x})"
                "    { const uint32_t _gx_src = " ++ bytes ++ "; if (_gx_src > " ++ n ++
                  "u) gx_ptx_trap_op(\"cp.async.src_size_exceeds_cp_size\"); gx_ptx_memcpy(" ++ dst ++ ", " ++
                  parseAddr a ++ ", _gx_src, 1); memset((char*)(uintptr_t)(" ++ dst ++ ") + _gx_src, 0, " ++ n ++
                  "u - _gx_src); }"
              | _ => trapOp
          | _ => trapOp
        else
          -- PTX lists state spaces in destination/source order.  Matching
          -- their ordered projection avoids confusing the two legal bulk
          -- directions merely because both contain `shared::cta`.
          let spaces := toks.filter (fun t => t == "global" || t == "shared" || t.startsWith "shared::")
          let bulk := toks.contains "bulk"
          let syntaxOK :=
            if bulk then
              toks == ["cp", "async", "bulk", "shared::cta", "global", "mbarrier::complete_tx::bytes"] ||
              toks == ["cp", "async", "bulk", "shared::cta", "global",
                "mbarrier::complete_tx::bytes", "L2::cache_hint"] ||
              toks == ["cp", "async", "bulk", "global", "shared::cta", "bulk_group"] ||
              toks == ["cp", "async", "bulk", "global", "shared::cta", "bulk_group",
                "L2::cache_hint"]
            else
              toks == ["cp", "async", "ca", "shared", "global"] ||
              toks == ["cp", "async", "ca", "shared::cta", "global"] ||
              toks == ["cp", "async", "cg", "shared", "global"] ||
              toks == ["cp", "async", "cg", "shared::cta", "global"]
          let sizedCopy (d a n sourceSize : String) :=
            if !bulk && ["4", "8", "16"].contains n &&
                (!toks.contains "cg" || n == "16") then
              match sourceSize.toNat? with
              | some sourceBytes =>
                if sourceBytes > n.toNat! then trapOp else
                  let dst := sharedPtr (parseAddr d regs)
                  s!"    gx_ptx_memcpy({dst}, {parseAddr a}, {sourceBytes}u, 1);" ++
                    s!" memset((char*)(uintptr_t)({dst}) + {sourceBytes}u, 0, {n.toNat! - sourceBytes}u);"
              | none => trapOp
            else trapOp
          -- General bulk route (cluster destinations, multicast, .ignore_oob,
          -- .cp_mask, .weak/.relaxed.scope.b128) through emu/ptx/bulk_copy.cc.
          let bulkRoute : String :=
            let quals := (toks.drop 3).filter (fun t => !semScope.contains t)
            let flags := quals.drop 3
            let known := flags.all (fun t => ["multicast::cluster", "L2::cache_hint", "ignore_oob", "cp_mask"].contains t)
            let hint := if flags.contains "L2::cache_hint" then 1 else 0
            if !bulk || toks.take 3 != ["cp", "async", "bulk"] || !known then trapOp
            else match quals.take 3, ops with
              | [dst, "global", "mbarrier::complete_tx::bytes"], d :: src :: n :: more =>
                if dst != "shared::cta" && dst != "shared::cluster" then trapOp
                else
                  let oob := flags.contains "ignore_oob"
                  let multicast := flags.contains "multicast::cluster"
                  if (oob && (dst != "shared::cta" || multicast)) || (multicast && dst != "shared::cluster") || flags.contains "cp_mask" then trapOp
                  else
                    let (il, ir, more) := if oob then (more.getD 0 "", more.getD 1 "", more.drop 2) else ("0", "0", more)
                    match more with
                    | mbar :: tail =>
                      let mask := if multicast then tail.headD "" else "0"
                      if tail.length != (if multicast then 1 else 0) + hint || il.isEmpty || ir.isEmpty || mask.isEmpty then trapOp
                      else s!"    gx_bulk_g2s((uint32_t)({parseAddr d regs}), (uint64_t)({parseAddr src regs}), (uint32_t)({oC n}), " ++
                        s!"(uint32_t)({parseAddr mbar regs}), {if multicast then "1" else "0"}u, (uint32_t)({oC mask}), (uint32_t)({oC il}), (uint32_t)({oC ir}));"
                    | [] => trapOp
              | ["shared::cluster", "shared::cta", "mbarrier::complete_tx::bytes"], [d, src, n, mbar] =>
                if !flags.isEmpty then trapOp
                else s!"    gx_bulk_s2s((uint32_t)({parseAddr d regs}), (uint32_t)({parseAddr src regs}), (uint32_t)({oC n}), (uint32_t)({parseAddr mbar regs}));"
              | ["global", "shared::cta", "bulk_group"], d :: src :: n :: more =>
                let cpMask := flags.contains "cp_mask"
                if flags.contains "multicast::cluster" || flags.contains "ignore_oob" ||
                   more.length != hint + (if cpMask then 1 else 0) then trapOp
                else
                  let byteMask := if cpMask then oC (more.getLastD "0") else "0xffffu"
                  s!"    gx_bulk_s2g((uint64_t)({parseAddr d regs}), (uint32_t)({parseAddr src regs}), (uint32_t)({oC n}), (uint32_t)({byteMask}));"
              | _, _ => trapOp
          if !syntaxOK || bulk then bulkRoute
          else match spaces, args with
          | [dstSpace, "global"], [d, a, n] =>
            if !bulk && (dstSpace == "shared" || dstSpace == "shared::cta") &&
                ["4", "8", "16"].contains n && (!toks.contains "cg" || n == "16") then
              let dst := sharedPtr (parseAddr d regs)
              "    " ++ CommTemplate.copyBytes.print "" "" dst (parseAddr a) (oC n) "1" ""
            else trapOp
          | ["shared::cta", "global"], [d, a, n, mbar] =>
            if bulk && toks.contains "mbarrier::complete_tx::bytes" &&
                !toks.contains "bulk_group" && !toks.contains "L2::cache_hint" then
              let dst := sharedPtr (parseAddr d regs)
              "    " ++ CommTemplate.copyBytes.print "" "" dst (parseAddr a) (oC n) "1" "" ++
                s!" gx_mbarrier_complete_tx({sharedPtr (parseAddr mbar regs)}, (uint32_t)({oC n}));"
            else sizedCopy d a n mbar
          | ["shared", "global"], [d, a, n, sourceSize] =>
            sizedCopy d a n sourceSize
          | ["shared::cta", "global"], [d, a, n, mbar, _cachePolicy] =>
            if bulk && toks.contains "mbarrier::complete_tx::bytes" &&
                !toks.contains "bulk_group" && toks.contains "L2::cache_hint" then
              let dst := sharedPtr (parseAddr d regs)
              "    " ++ CommTemplate.copyBytes.print "" "" dst (parseAddr a) (oC n) "1" "" ++
                s!" gx_mbarrier_complete_tx({sharedPtr (parseAddr mbar regs)}, (uint32_t)({oC n}));"
            else trapOp
          | ["global", "shared::cta"], [d, a, n] =>
            if bulk && toks.contains "bulk_group" &&
                !toks.contains "mbarrier::complete_tx::bytes" &&
                !toks.contains "L2::cache_hint" then
              let src := sharedPtr (parseAddr a regs)
              "    " ++ CommTemplate.copyBytes.print "" "" (parseAddr d) src (oC n) "0" ""
            else trapOp
          | ["global", "shared::cta"], [d, a, n, _cachePolicy] =>
            if bulk && toks.contains "bulk_group" &&
                !toks.contains "mbarrier::complete_tx::bytes" &&
                toks.contains "L2::cache_hint" then
              let src := sharedPtr (parseAddr a regs)
              "    " ++ CommTemplate.copyBytes.print "" "" (parseAddr d) src (oC n) "0" ""
            else trapOp
          -- shared::cluster needs remote-CTA address/runtime support; source
          -- sizing, masks, unsupported cache-policy shapes, and typed relaxed
          -- copies likewise cannot reuse a plain memcpy route.
          | _, _ => trapOp
      /- PTX definition: `mbarrier` instructions track arrival and test/wait completion state across participating threads. -/
      /- Lean modeling: the supported CTA subset delegates arrival/transaction counts and phase tests to per-launch runtime state; unmodeled forms trap and runtime synchronization remains trusted. -/
      | "mbarrier" =>
        -- Canonical form: mbarrier.<op>[.expect_tx][.parity][.sem][.scope][.space].b64.
        -- Semantics/scope qualifiers order memory only; GX executes fibers
        -- cooperatively on one pthread, so they do not change values.
        let addr (a : String) := sharedPtr (parseAddr a regs)
        let quals := (toks.drop 1).filter (fun t =>
          !["release", "acquire", "relaxed", "cta", "cluster", "shared", "shared::cta",
            "shared::cluster", "b64"].contains t)
        let spaceOK := toks.getLast? == some "b64" &&
          (toks.contains "shared" || toks.contains "shared::cta" || toks.contains "shared::cluster")
        let localOnly := !clusterShared
        if !spaceOK then trapOp
        else if quals == ["init"] && localOnly then
          match args with
          | [a, count] => s!"    gx_mbarrier_init({addr a}, (uint32_t)({oC count}));"
          | _ => trapOp
        else if quals == ["arrive"] || quals == ["arrive", "expect_tx"] then
          let result (d a count bytes : String) :=
            let call := s!"gx_mbarrier_arrive({addr a}, (uint32_t)({count}), (uint32_t)({bytes}))"
            if d == "_" then s!"    (void){call};"
            -- A remote (shared::cluster) arrival returns no state token.
            else if clusterShared then trapOp
            else s!"    {dest d} = {call};"
          match args with
          | [d, a] => if toks.contains "expect_tx" then trapOp else result d a "1" "0"
          | [d, a, n] => if toks.contains "expect_tx" then result d a "1" (oC n)
                        else result d a (oC n) "0"
          | _ => trapOp
        else if quals == ["expect_tx"] then
          match args with
          | [a, n] => s!"    gx_mbarrier_expect_tx({addr a}, (uint32_t)({oC n}));"
          | _ => trapOp
        else if quals == ["complete_tx"] then
          match args with
          | [a, n] => s!"    gx_mbarrier_complete_tx({addr a}, (uint32_t)({oC n}));"
          | _ => trapOp
        else if (quals == ["try_wait", "parity"] || quals == ["try_wait"] ||
                 quals == ["test_wait", "parity"] || quals == ["test_wait"]) && localOnly then
          match args with
          | [d, a, phase] | [d, a, phase, _] =>
            if d == "_" then trapOp
            else s!"    {dest d} = gx_mbarrier_test({addr a}, (uint64_t)({oC phase}), {if toks.contains "parity" then "1" else "0"});"
          | _ => trapOp
        else trapOp
      /- PTX definition: `mapa` maps a shared-memory address of the executing CTA to the corresponding address in the cluster CTA of the given rank. -/
      /- Lean modeling: the shared::cluster form returns GX's rank-tagged 32/64-bit window address and the generic form returns a host pointer, both through runtime helpers covered by `trusted:cluster-runtime`; other spellings trap. -/
      | "mapa" =>
        match args with
        | [d, a, r] =>
          if toks == ["mapa", "shared::cluster", "u32"] || toks == ["mapa", "shared::cluster", "u64"] then
            s!"    {dest d} = gx_cluster_mapa((uint32_t)({oC a}), (uint32_t)({oC r}));"
          else if toks == ["mapa", "u64"] then
            s!"    {dest d} = gx_cluster_mapa_generic((uint64_t)({oC a}), (uint32_t)({oC r}));"
          else trapOp
        | _ => trapOp
      /- PTX definition: `prefetch`/`prefetchu` bring data or a tensor map into a cache level; they have no architecturally visible result. -/
      /- Lean modeling: every recognized prefetch spelling lowers to an empty statement because GX models no caches; malformed operand counts trap. -/
      | "prefetch" | "prefetchu" =>
        if args.length == 1 then ";" else trapOp
      /- PTX definition: `tcgen05` instructions allocate and access Tensor Memory, issue asynchronous 5th-generation MMAs, and signal their completion through mbarriers. -/
      /- Lean modeling: a closed spelling allowlist lowers each supported form to `trusted:tensor-memory` runtime helpers that execute at issue; unsupported shapes, kinds and modifiers trap. -/
      | "tcgen05" =>
        let sub := toks.getD 1 ""
        let group := if toks.contains "cta_group::2" then "2" else "1"
        let groupTok1 (ts : List String) := (ts.filter (fun t => t.startsWith "cta_group::")).length == 1
        let groupOK := (toks.filter (fun t => t.startsWith "cta_group::")).length ≤ 1 &&
          !toks.any (fun t => t.startsWith "cta_group::" && t != "cta_group::1" && t != "cta_group::2")
        if !groupOK then trapOp
        else if sub == "alloc" then
          if toks.filter (fun t => !t.startsWith "cta_group::") == ["tcgen05", "alloc", "sync", "aligned", "shared::cta", "b32"] then
            match args with
            | [d, n] => s!"    gx_tcgen05_alloc((uint32_t)({parseAddr d regs}), (uint32_t)({oC n}), {group}u);"
            | _ => trapOp
          else trapOp
        else if sub == "dealloc" then
          if toks.filter (fun t => !t.startsWith "cta_group::") == ["tcgen05", "dealloc", "sync", "aligned", "b32"] then
            match args with
            | [a, n] => s!"    gx_tcgen05_dealloc((uint32_t)({oC a}), (uint32_t)({oC n}), {group}u);"
            | _ => trapOp
          else trapOp
        else if sub == "relinquish_alloc_permit" then
          if toks.filter (fun t => !t.startsWith "cta_group::") == ["tcgen05", "relinquish_alloc_permit", "sync", "aligned"] &&
             args.isEmpty then ";" else trapOp
        else if sub == "wait::ld" || sub == "wait::st" then
          if toks.drop 2 == ["sync", "aligned"] && args.isEmpty then ";" else trapOp
        else if sub == "fence::before_thread_sync" || sub == "fence::after_thread_sync" then
          if toks.length == 2 && args.isEmpty then ";" else trapOp
        else if sub == "commit" then
          let mods := toks.filter (fun t => !t.startsWith "cta_group::")
          let multicast := mods.contains "multicast::cluster"
          let clusterSpace := mods.contains "shared::cluster"
          let okMods := mods.drop 2 |>.all (fun t => ["mbarrier::arrive::one", "shared::cluster",
                          "multicast::cluster", "b64"].contains t)
          if !okMods || !mods.contains "mbarrier::arrive::one" || mods.getLast? != some "b64" then trapOp
          else match args, multicast with
            | [m], false =>
              if clusterSpace then s!"    gx_tcgen05_commit((uint32_t)({parseAddr m regs}), 0u, 0u);"
              else s!"    (void)gx_mbarrier_arrive((uint64_t)({parseAddr m regs}), 1u, 0u);"
            | [m, mask], true =>
              if clusterSpace then s!"    gx_tcgen05_commit((uint32_t)({parseAddr m regs}), 1u, (uint32_t)({oC mask}));"
              else trapOp
            | _, _ => trapOp
        else if sub == "ld" && toks.getD 2 "" == "red" then
          -- tcgen05.ld.red.sync.aligned.{32x32b|16x32bx2}.xN.{min|max}[.abs][.NaN].f32 | .{u32|s32}
          --   {r...}, redval, [taddr] [, immHalfSplitoff]
          let shape := toks.getD 5 ""
          let rep := toks.getD 6 ""
          let n := (rep.drop 1).toNat?.getD 0
          let qs := toks.drop 7
          let op := if qs.contains "max" then 1 else 0
          let ty := if qs.contains "f32" then some 0 else if qs.contains "u32" then some 1 else if qs.contains "s32" then some 2 else none
          let absQ := qs.contains "abs"
          let nanQ := qs.contains "NaN"
          let qsOK := qs.all (fun t => ["min", "max", "abs", "NaN", "f32", "u32", "s32"].contains t) &&
            (qs.filter (fun t => t == "min" || t == "max")).length == 1 && qs.length == 2 + (if absQ then 1 else 0) + (if nanQ then 1 else 0) &&
            ((!absQ && !nanQ) || ty == some 0)
          let code := if shape == "32x32b" then some 0 else if shape == "16x32bx2" then some 4 else none
          match code, ty with
          | some c, some t =>
            if toks.take 5 != ["tcgen05", "ld", "red", "sync", "aligned"] || !qsOK || !rep.startsWith "x" ||
               ![2, 4, 8, 16, 32, 64, 128].contains n then trapOp
            else
              let (body, imm) := if c == 4 then (args.dropLast, args.getLastD "0") else (args, "0")
              match body.reverse with
              | addr :: red :: rs =>
                let dsts := rs.reverse
                if dsts.length != n || dsts.contains "_" || red == "_" then trapOp
                else
                  let writes := String.intercalate " " (dsts.enum.map (fun (i, r) => s!"gx_movb({dest r}, (uint32_t)_gx_t[{i}]);"))
                  "    { uint32_t _gx_t[" ++ toString n ++ "]; uint32_t _gx_r = gx_tcgen05_ld_red(" ++ toString c ++ "u, (uint32_t)(" ++
                    parseAddr addr regs ++ "), " ++ toString n ++ "u, (uint32_t)(" ++ oC imm ++ "), _gx_t, " ++ toString op ++ "u, " ++
                    toString t ++ "u, " ++ (if absQ then "1u" else "0u") ++ ", " ++ (if nanQ then "1u" else "0u") ++ "); " ++
                    writes ++ " gx_movb(" ++ dest red ++ ", _gx_r); }"
              | _ => trapOp
          | _, _ => trapOp
        else if sub == "ld" || sub == "st" then
          -- tcgen05.{ld,st}.sync.aligned.<shape>.x<N>[.pack::16b|.unpack::16b].b32
          let shape := toks.getD 4 ""
          let rep := toks.getD 5 ""
          let n := (rep.drop 1).toNat?.getD 0
          let tail := toks.drop 6
          let pack := tail == ["pack::16b", "b32"] || tail == ["unpack::16b", "b32"]
          let packOK := tail == ["b32"] || (sub == "ld" && tail == ["pack::16b", "b32"]) ||
                        (sub == "st" && tail == ["unpack::16b", "b32"])
          let shapeInfo : Option (Nat × Nat) := match shape with
            | "32x32b" => some (0, 1) | "16x64b" => some (1, 1) | "16x128b" => some (2, 2)
            | "16x256b" => some (3, 4) | "16x32bx2" => some (4, 1) | _ => none
          let validN := [1, 2, 4, 8, 16, 32, 64, 128].contains n
          match shapeInfo with
          | none => trapOp
          | some (code, mult) =>
            let nregs := n * mult
            let hasImm := code == 4
            if toks.take 4 != ["tcgen05", sub, "sync", "aligned"] || !rep.startsWith "x" || !validN || !packOK then trapOp
            else if sub == "ld" then
              let (body, imm) := if hasImm then (args.dropLast, args.getLastD "0") else (args, "0")
              match body.reverse with
              | a :: rs =>
                let dsts := rs.reverse
                if dsts.length != nregs || dsts.contains "_" then trapOp
                else
                  let writes := String.intercalate " " (dsts.enum.map (fun (i, r) => s!"gx_movb({dest r}, (uint32_t)_gx_t[{i}]);"))
                  "    { uint32_t _gx_t[" ++ toString nregs ++ "]; gx_tcgen05_ld(" ++ toString code ++ "u, (uint32_t)(" ++
                    parseAddr a regs ++ "), " ++ toString nregs ++ "u, " ++ (if pack then "1u" else "0u") ++ ", (uint32_t)(" ++
                    oC imm ++ "), _gx_t); " ++ writes ++ " }"
              | _ => trapOp
            else
              match args with
              | a :: rest =>
                let (srcs, imm) := if hasImm then
                    match rest with
                    | i :: ss => (ss, i)
                    | [] => ([], "0")
                  else (rest, "0")
                if srcs.length != nregs then trapOp
                else
                  let vals := String.intercalate ", " (srcs.map (fun r => s!"gx_bits32({oC r})"))
                  "    { const uint32_t _gx_t[" ++ toString nregs ++ "] = {" ++ vals ++ "}; gx_tcgen05_st(" ++ toString code ++
                    "u, (uint32_t)(" ++ parseAddr a regs ++ "), " ++ toString nregs ++ "u, " ++ (if pack then "1u" else "0u") ++
                    ", (uint32_t)(" ++ oC imm ++ "), _gx_t); }"
              | _ => trapOp
        else if sub == "cp" then
          -- tcgen05.cp.cta_group::N.<shape>[.warpx..][.b8x16.<src>] [taddr], s-desc
          let mods := (toks.drop 2).filter (fun t => !t.startsWith "cta_group::")
          let shapeCode : Option Nat := match mods.take 2 with
            | ["128x256b", _] | ["128x256b"] => some 0
            | ["128x128b", _] | ["128x128b"] => some 1
            | ["4x256b", _] | ["4x256b"] => some 2
            | ["64x128b", "warpx2::02_13"] => some 3
            | ["64x128b", "warpx2::01_23"] => some 4
            | ["32x128b", "warpx4"] => some 5
            | _ => none
          let shapeLen := if shapeCode == some 3 || shapeCode == some 4 || shapeCode == some 5 then 2 else 1
          let decomp := mods.drop shapeLen
          let decompCode : Option Nat := match decomp with
            | [] => some 0 | ["b8x16", "b4x16_p64"] => some 1 | ["b8x16", "b6x16_p32"] => some 2 | _ => none
          match shapeCode, decompCode, args with
          | some sc, some dc, [a, d] =>
            s!"    gx_tcgen05_cp({sc}u, {dc}u, {group}u, (uint32_t)({parseAddr a regs}), (uint64_t)({oC d}));"
          | _, _, _ => trapOp
        else if sub == "mma" then
          -- tcgen05.mma{.sp}{.ws}.cta_group::N.kind::K[.block_scale[.blockB|.scale_vec::SX]]
          --   [.collector::{a|bN}::op][.ashift]
          --   [d], a, b, [[sp-meta],] idesc, [[sfa], [sfb],] [{disable-output-lane x4|x8},]
          --   enable-input-d [, scale-input-d | , zero-column-mask-desc (.ws)]
          -- Collector-buffer qualifiers are cache hints: for valid programs
          -- reading the operand again gives identical results.
          let sp := toks.getD 2 "" == "sp" || toks.getD 3 "" == "sp"
          let ws := toks.getD 2 "" == "ws" || toks.getD 3 "" == "ws"
          let mods := (toks.drop 2).filter (fun t => !t.startsWith "cta_group::" && t != "sp" && t != "ws")
          let kindTok := (mods.headD "").drop 6
          let kindCode := match kindTok with
            | "f16" => some 0 | "tf32" => some 1 | "f8f6f4" => some 2 | "i8" => some 3
            | "mxf8f6f4" => some 4 | "mxf4" => some 5 | "mxf4nvf4" => some 6 | _ => none
          let scaled := mods.contains "block_scale"
          let isScaledKind := kindCode == some 4 || kindCode == some 5 || kindCode == some 6
          let block := if mods.contains "block16" || mods.contains "scale_vec::4X" then 16
            else if mods.contains "block32" || mods.contains "scale_vec::1X" || mods.contains "scale_vec::2X" then 32 else 0
          let collector (t : String) := t.startsWith "collector::" &&
            (if ws then ["b0", "b1", "b2", "b3"].any (fun bn => ["fill", "use", "lastuse", "discard"].any (fun o => t == s!"collector::{bn}::{o}"))
             else ["fill", "use", "lastuse", "discard"].any (fun o => t == s!"collector::a::{o}"))
          let ashift := mods.contains "ashift"
          let extraOK := (mods.drop 1).all (fun t => ["block_scale", "block16", "block32",
                            "scale_vec::1X", "scale_vec::2X", "scale_vec::4X", "ashift"].contains t || collector t) &&
            (mods.filter collector).length ≤ 1 &&
            !(ashift && (ws || mods.any (fun t => t == "collector::a::use" || t == "collector::a::fill")))
          let groupTok := toks.filter (fun t => t.startsWith "cta_group::")
          if !(mods.headD "").startsWith "kind::" || kindCode.isNone || !extraOK || scaled != isScaledKind ||
             (ws && (scaled || groupTok != ["cta_group::1"])) || groupTok.length != 1 then trapOp
          else match args with
            | d :: a :: b :: more =>
              let aTmem := a.startsWith "["
              let (meta, more) := if sp then
                  match more with
                  | m :: r => (s!"(uint32_t)({parseAddr m regs})", r)
                  | [] => ("", [])
                else ("0u", more)
              match more with
              | idesc :: more =>
                let (sfa, sfb, rest) := if scaled then
                    match more with
                    | x :: y :: r => (s!"(uint32_t)({parseAddr x regs})", s!"(uint32_t)({parseAddr y regs})", r)
                    | _ => ("", "", [])
                  else ("0u", "0u", more)
                let nMask := if group == "2" then 8 else 4
                -- Flattened tail: non-scaled forms are [mask x nMask,] en [, scale]; .ws is en [, zcm].
                let (masks, after) := if !scaled && !ws && rest.length ≥ nMask + 1 then (rest.take nMask, rest.drop nMask)
                  else ([], rest)
                let pred (p : String) := if p.startsWith "!" then s!"(!({oC (p.drop 1)}))" else s!"({oC p})"
                let aExpr := if aTmem then s!"(uint64_t)({parseAddr a regs})" else s!"(uint64_t)({oC a})"
                let emit (en scale zcm : String) (flagsExtra : Nat) : String :=
                  let flags := (if sp then 1 else 0) + (if ws then 2 else 0) + (if ashift then 4 else 0) + flagsExtra
                  let maskDecl := if masks.isEmpty then "" else
                    "const uint32_t _gx_m[" ++ toString nMask ++ "] = {" ++
                      String.intercalate ", " (masks.map (fun m => s!"(uint32_t)({oC m})")) ++ "}; "
                  let maskArg := if masks.isEmpty then "nullptr, 0u" else s!"_gx_m, {nMask}u"
                  "    { " ++ maskDecl ++ s!"gx_tcgen05_mma({kindCode.getD 0}u, {group}u, (uint32_t)({parseAddr d regs}), {aExpr}, {if aTmem then "1" else "0"}u, " ++
                    s!"(uint64_t)({oC b}), (uint32_t)({oC idesc}), (uint32_t){pred en}, {sfa}, {sfb}, {block}u, {maskArg}, " ++
                    s!"(uint32_t)({scale}), {meta}, {flags}u, (uint64_t)({zcm})); " ++ "}"
                if sfa.isEmpty || meta.isEmpty || (ashift && !aTmem) then trapOp
                else match after with
                  | [en] => emit en "0" "0" 0
                  | [en, x] =>
                    if ws then emit en "0" (oC x) 8
                    else if scaled then trapOp
                    else emit en (oC x) "0" 0
                  | _ => trapOp
              | _ => trapOp
            | _ => trapOp
        else if sub == "shift" then
          -- tcgen05.shift.cta_group::N.down [taddr]  (either qualifier order)
          if (toks.drop 2).filter (fun t => !t.startsWith "cta_group::") == ["down"] && groupTok1 toks then
            match args with
            | [a] => s!"    gx_tcgen05_shift((uint32_t)({parseAddr a regs}), {group}u);"
            | _ => trapOp
          else trapOp
        else trapOp
      /- PTX definition: `tensormap.replace` rewrites one field of a 128-byte tensor map in global or shared memory; `tensormap.cp_fenceproxy` copies a tensor map from shared to global memory and fences it for the tensormap proxy. -/
      /- Lean modeling: field rewrites go to the trusted TMA runtime's GX descriptor (PTX Table 33 encodings); the copy is a 128-byte move, and the proxy fence is subsumed by GX reading the descriptor at TMA issue. Other spellings trap. -/
      | "tensormap" =>
        if toks.take 3 == ["tensormap", "replace", "tile"] then
          let field := toks.getD 3 ""
          let fieldCode : Option Nat := match field with
            | "global_address" => some 0 | "rank" => some 1 | "box_dim" => some 2 | "global_dim" => some 3
            | "global_stride" => some 4 | "element_stride" => some 5 | "elemtype" => some 6
            | "interleave_layout" => some 7 | "swizzle_mode" => some 8 | "swizzle_atomicity" => some 9
            | "fill_mode" => some 10 | _ => none
          let space := toks.getD 4 ""
          let wide := field == "global_address" || field == "global_stride"
          let tyOK := toks.drop 5 == ["b1024", if wide then "b64" else "b32"]
          let obj (a : String) := if space == "shared::cta" then s!"((uint64_t)(uintptr_t)(_gx_shm + (({parseAddr a regs}) & 0xFFFFFFu)))"
            else s!"(uint64_t)({parseAddr a regs})"
          match fieldCode, args with
          | some fc, [a, v] =>
            if !tyOK || (space != "global" && space != "shared::cta") || [2, 3, 4, 5].contains fc then trapOp
            else s!"    gx_tensormap_replace({obj a}, {fc}u, 0u, (uint64_t)({oC v}));"
          | some fc, [a, ord, v] =>
            if !tyOK || (space != "global" && space != "shared::cta") || ![2, 3, 4, 5].contains fc then trapOp
            else s!"    gx_tensormap_replace({obj a}, {fc}u, (uint32_t)({oC ord}), (uint64_t)({oC v}));"
          | _, _ => trapOp
        else if toks.take 2 == ["tensormap", "cp_fenceproxy"] then
          -- tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.<scope>.sync.aligned [g], [s], 128
          let q := toks.drop 2
          let scopeOK := ["cta", "cluster", "gpu", "sys"].contains (q.getD 4 "")
          if q.take 4 != ["global", "shared::cta", "tensormap::generic", "release"] || !scopeOK ||
             q.drop 5 != ["sync", "aligned"] then trapOp
          else match args with
            | [g, sm, n] =>
              if n != "128" then trapOp
              else s!"    memcpy((void*)(uintptr_t)({parseAddr g regs}), (const void*)(_gx_shm + (({parseAddr sm regs}) & 0xFFFFFFu)), 128); __atomic_thread_fence(__ATOMIC_SEQ_CST);"
            | _ => trapOp
        else trapOp
      /- PTX definition: `clusterlaunchcontrol.try_cancel` asynchronously requests cancellation of a not-yet-launched cluster and writes an opaque 16-byte response; `query_cancel` decodes it. -/
      /- Lean modeling: the runtime cancels the next pending cluster in launch order (or reports failure) and delivers its GX-encoded response with mbarrier transaction completion; queries decode that encoding; other spellings trap. -/
      | "clusterlaunchcontrol" =>
        let lo (r : String) := s!"(uint64_t)({oC r})"
        let hi (r : String) := s!"(uint64_t)(((unsigned __int128)({oC r})) >> 64)"
        if toks.take 2 == ["clusterlaunchcontrol", "try_cancel"] then
          let mods := toks.drop 2
          let multicast := mods.contains "multicast::cluster::all"
          if mods.filter (· != "multicast::cluster::all") ==
               ["async", "shared::cta", "mbarrier::complete_tx::bytes", "b128"] then
            match args with
            | [r, m] => s!"    gx_clc_try_cancel((uint32_t)({parseAddr r regs}), (uint32_t)({parseAddr m regs}), {if multicast then "1" else "0"}u);"
            | _ => trapOp
          else trapOp
        else if toks == ["clusterlaunchcontrol", "query_cancel", "is_canceled", "pred", "b128"] then
          match args with
          | [d, r] => s!"    {dest d} = gx_clc_query({lo r}, {hi r}, 0u);"
          | _ => trapOp
        else if toks == ["clusterlaunchcontrol", "query_cancel", "get_first_ctaid", "v4", "b32", "b128"] then
          match args with
          | [x, y, z, w, r] =>
            let put (d : String) (i : Nat) := if d == "_" then "" else s!" {dest d} = gx_clc_query({lo r}, {hi r}, {i}u);"
            "   " ++ put x 1 ++ put y 2 ++ put z 3 ++ put w 4
          | _ => trapOp
        else if toks.take 3 == ["clusterlaunchcontrol", "query_cancel", "get_first_ctaid::x"] ||
                toks.take 3 == ["clusterlaunchcontrol", "query_cancel", "get_first_ctaid::y"] ||
                toks.take 3 == ["clusterlaunchcontrol", "query_cancel", "get_first_ctaid::z"] then
          let i := if toks.contains "get_first_ctaid::x" then 1 else if toks.contains "get_first_ctaid::y" then 2 else 3
          match args, toks.drop 3 with
          | [d, r], ["b32", "b128"] => s!"    {dest d} = gx_clc_query({lo r}, {hi r}, {i}u);"
          | _, _ => trapOp
        else trapOp
      /- PTX definition: `nanosleep` suspends execution for an approximate interval before resuming. -/
      /- Lean modeling: a scheduler yield preserves progress intent rather than timing and is classified as structural runtime plumbing, not exact duration semantics. -/
      | "nanosleep" => "    gx_cpu_kernel_yield();"
      /- PTX definition: barrier instructions synchronize named participants and optional reductions combine predicates. -/
      /- Lean modeling: helper selection and predicate negation preserve operand intent, but runtime synchronization/reduction correspondence remains structural and `gap:sync-runtime-limits`. -/
      | "barrier" | "bar" =>
        if toks.contains "warp" then
          match args with | [m] => s!"    __syncwarp_mask((unsigned){oC m});" | _ => trapOp
        else if toks.contains "cluster" then
          -- barrier.cluster.arrive[.release|.relaxed][.aligned] /
          -- barrier.cluster.wait[.acquire][.aligned]: split-phase barrier over
          -- every thread of every CTA in the cluster.
          let extra := (toks.drop 3).filter (fun t => !["release", "relaxed", "acquire", "aligned"].contains t)
          if !args.isEmpty || !extra.isEmpty then trapOp
          else if toks.take 3 == ["barrier", "cluster", "arrive"] && !toks.contains "acquire" then
            "    gx_cluster_barrier_arrive();"
          else if toks.take 3 == ["barrier", "cluster", "wait"] &&
                  !toks.contains "release" && !toks.contains "relaxed" then
            "    gx_cluster_barrier_wait();"
          else trapOp
        else if toks.contains "red" then
          let fn := if toks.contains "or" then "gx_barrier_red_or" else if toks.contains "and" then "gx_barrier_red_and" else "gx_barrier_red_popc"
          let emit (d nm cnt p : String) :=
            let pv := oC (p.replace "!" "")
            let pp := if p.startsWith "!" then s!"(!{pv})" else pv
            s!"    {dest d} = {fn}((int){oC nm}, (int){oC cnt}, (int){pp});"
          if !toks.any (fun t => ["or", "and", "popc"].contains t) then trapOp
          else match args with
            | [d, nm, p] => emit d nm "0" p
            | [d, nm, cnt, p] => emit d nm cnt p
            | _ => trapOp
        else match args with
             | [id, cnt] => s!"    barrier_sync((int){oC id}, (int){oC cnt});"
             | [id] => s!"    barrier_sync((int){oC id});"
             | _ => trapOp
      -- ==================================================================
      -- BIT MANIPULATION: FUNNEL, EXTRACT, INSERT, AND PERMUTE
      -- ==================================================================
      /- PTX definition: `shf.l/r.wrap/clamp.b32` selects bits from two concatenated words using a wrapped or clamped amount. -/
      /- Lean modeling: funnel values and amount normalization have shallow equalities; their helper spelling/compiler execution remains `trusted:template-print`. -/
      | "shf"  =>
        -- Funnel values are covered by shallow CBits equalities; the printed
        -- helper/compiler route remains `trusted:template-print`.
        match args with
        | d :: a :: b :: c :: _ =>
          s!"    {dest d} = {printShf (toks.contains "r") (toks.contains "wrap") (oC a) (oC b) (oC c)};"
        | _ => s!"    /* malformed: {s} */"
      /- PTX definition: `bfe` extracts a field and zero- or sign-extends it according to the declared type. -/
      /- Lean modeling: `bfeU_spec` and `bfeS_spec` cover normalized controls plus zero/sign padding; helper spelling and compiler execution remain the trusted print boundary. -/
      | "bfe"  => match args with
                  | [d, a, p, l] =>
                    -- Both forms have per-bit value theorems
                    -- (CBits.bfeU_spec / CBits.bfeS_spec); spelling is trusted.
                    if d == "_" || !(["u32", "s32", "u64", "s64"].contains tyTok) ||
                        toks != ["bfe", tyTok] then trapOp
                    else
                      s!"    {dest d} = {printBfe (tyTok.startsWith "s") (tyTok.endsWith "64") (oC a) (oC p) (oC l)};"
                  | _ => trapOp
      | "bmsk" =>
        if toks != ["bmsk", "clamp", "b32"] && toks != ["bmsk", "wrap", "b32"] then trapOp
        else match args with
        | [d, a, b] =>
          if d == "_" then trapOp
          else s!"    gx_movb({dest d}, gx_bmsk32({bitOperand regs "b32" a}, {bitOperand regs "b32" b}, {if toks.contains "wrap" then "true" else "false"}));"
        | _ => trapOp
      /- PTX definition: `bfi` inserts a low-bit field from one source into a base value at the selected position. -/
      /- Lean modeling: insertion has a shallow per-bit value theorem; the helper spelling and compiler execution remain `trusted:template-print`. -/
      | "bfi"  => match args with
                  | d :: ins :: base :: p :: l :: _ =>
                    -- Shallow value theorem: CBits.bfi_spec (ISA per-bit rule);
                    -- helper printing/compiler execution remains trusted.
                    s!"    {dest d} = {printBfi (tyTok.endsWith "64") (oC ins) (oC base) (oC p) (oC l)};"
                  | _ => s!"    /* malformed: {s} */"
      /- PTX definition: `prmt.b32` selects output bytes from two inputs under generic or specialized mode rules. -/
      /- Lean modeling: generic permutation has shallow value equality and trusted helper printing; specialized modes are rejected instead of inheriting the generic proof. -/
      | "prmt" =>
        -- Generic mode has shallow value equality (CBits.prmt_equiv) and a
        -- trusted printer/compiler link; other modes trap loudly.
        if toks.any (fun t => t == "f4e" || t == "b4e" || t == "rc8"
                       || t == "ecl" || t == "ecr" || t == "rc16") then
          s!"    gx_ptx_trap_op(\"{op}\");"
        else match args with
             | d :: a :: b :: c :: _ => s!"    {dest d} = {printPrmt (oC a) (oC b) (oC c)};"
             | _ => s!"    /* malformed: {s} */"
      /- PTX definition: `bfind` locates the most-significant non-sign bit and optionally returns a shift amount. -/
      /- Lean modeling: the arm preserves all selector flags and operand order, while runtime helper/compiler correctness remains the explicit `trusted:bfind` boundary. -/
      | "bfind" => match args with
                  | d :: a :: _ =>
                    let sh := if toks.contains "shiftamt" then "1" else "0"
                    let sg := if tyTok.startsWith "s" then "1" else "0"
                    s!"    {dest d} = {printBfind (tyTok.endsWith "64") (oC a) sh sg};"
                  | _ => s!"    /* malformed: {s} */"
      /- PTX definition: `trap` aborts execution and transfers control to the execution environment. -/
      /- Lean modeling: `gx_ptx_trap()` preserves unconditional abnormal termination at the supported boundary; exact environment behavior is intentionally classified as `trap`. -/
      | "trap" => "    gx_ptx_trap();"
      | _ => if head.startsWith "prototype" || head == "callseq" then ""
             -- cold paths (multimem/atom/NVLS) neither translator executes:
             -- emit a runtime trap, matching gxptx.py's lenient mode.
             else s!"    gx_ptx_trap_op(\"{op}\");"

      let floating := toks.any (fun t => ["f16", "f16x2", "bf16", "bf16x2", "f32", "f32x2", "f64"].contains t)
      let packed := toks.any (fun t => t.endsWith "x2")
      let scalarFlops := if floating && ["add", "sub", "mul", "fma", "mad", "div", "sqrt", "min", "max"].contains head
                         then (if head == "fma" || head == "mad" then 2 else 1) * (if packed then 2 else 1) else 0
      let shape := toks.find? (fun t => t.startsWith "m" && t.contains 'n' && t.contains 'k')
      let dims := shape.map (fun t => (t.replace "m" " " |>.replace "n" " " |>.replace "k" " ").splitOn " " |>.filterMap String.toNat?)
      let tensor := head == "mma" || (head == "wmma" && toks.contains "mma") || (head == "wgmma" && toks.contains "mma_async")
      let tensorWork := if tensor then dims.map (fun ds => 2 * ds.foldl (· * ·) 1 / (if head == "wgmma" then 128 else 32)) |>.getD 0 else 0
      let memory := head == "ld" || head == "st" || head == "atom" || head == "red"
      let vector := if toks.contains "v4" then 4 else if toks.contains "v2" then 2 else if toks.contains "v8" then 8 else 1
      let elemBytes := if tyTok.endsWith "128" then 16 else if tyTok.endsWith "64" then 8 else if tyTok.endsWith "16" then 2 else if tyTok.endsWith "8" then 1 else 4
      let matrixTransfer := head == "ldmatrix" || head == "stmatrix"
      let wmmaMemory := head == "wmma" && (toks.contains "load" || toks.contains "store")
      let matrixBytes := if matrixTransfer then 4 * (if toks.contains "x4" then 4 else if toks.contains "x2" then 2 else 1)
                         else if wmmaMemory then dims.map (fun ds =>
                           let m := ds.getD 0 0; let n := ds.getD 1 0; let k := ds.getD 2 0
                           elemBytes * (if toks.contains "a" then m*k else if toks.contains "b" then k*n else m*n) / 32) |>.getD 0 else 0
      let bytes := if memory then vector * elemBytes else matrixBytes
      let space := if isShared || matrixTransfer then 2 else if toks.contains "local" then 3 else if toks.any (fun t => t.startsWith "param") then 4 else if toks.contains "const" then 5 else if toks.contains "global" then 1 else 0
      let reads := if head == "ld" || head == "atom" || head == "red" || head == "ldmatrix" || (wmmaMemory && toks.contains "load") then bytes else 0
      let writes := if head == "st" || head == "atom" || head == "red" || head == "stmatrix" || (wmmaMemory && toks.contains "store") then bytes else 0
      let addr := if memory then (args.find? (fun a => a.startsWith "[")).map mkAddr |>.getD "0" else "0"
      -- Predicate and label recursion place these counters on executed paths.
      let resultType := (toks.filter (fun t => ["f16", "f16x2", "bf16", "bf16x2", "f32", "f32x2", "f64"].contains t)).headD ""
      let kind := if tensor then (if floating then 2 else 3) else if scalarFlops > 0 then (if resultType == "f64" then 4 else if resultType.startsWith "bf16" then 6 else if resultType.startsWith "f16" then 5 else 1) else 0
      -- Supported async-copy forms have semantic byte counters in the runtime.
      -- Ordinary cp.async has an immediate destination size and optional
      -- dynamic source size / ignore-source predicate; count zero-fill writes.
      let simpleCopy := head == "cp" && toks.getD 1 "" == "async" && ["ca", "cg"].contains (toks.getD 2 "")
      let copyBytes := if simpleCopy then (args.getD 2 "0").toNat?.getD 0 else 0
      let copyExtra := if toks.contains "L2::cache_hint" then (args.drop 3).dropLast else args.drop 3
      let copyReads := match copyExtra with
        | [x] => if regs.contains (x ++ "#pred") then s!"(({oC x}) ? 0u : {copyBytes}u)" else s!"(uint32_t)({oC x})"
        | _ => toString copyBytes
      let reads := if simpleCopy then copyReads else toString reads
      let writes := if simpleCopy then copyBytes else writes
      let space := if simpleCopy then 6 else space
      let addr := if simpleCopy then mkAddr (args.getD 1 "0") else addr
      let unmodeled := head == "multimem" || (wmmaMemory && matrixBytes == 0) || (head == "wgmma" && tensor && tensorWork == 0)
      let emitted := if (emitted.splitOn "/* malformed:").length > 1 then trapOp else emitted
      -- ld.acquire / st.release: host acquire/release fences give the ordering a
      -- cross-thread (multi-rank) poll/publish handshake needs; the access
      -- helpers are opaque calls, so polling loops re-read memory.
      let emitted :=
        if emitted.startsWith "    gx_ptx_trap" then emitted
        else if head == "ld" && toks.contains "acquire" then emitted ++ "\n    __atomic_thread_fence(__ATOMIC_ACQUIRE);"
        else if head == "st" && toks.contains "release" then "    __atomic_thread_fence(__ATOMIC_RELEASE);\n" ++ emitted
        else emitted
      -- Annotation calls are collector metadata, not kernel work. Parameter
      -- staging and PCV expression evaluation remain ordinary PTX operations.
      if !profile || emitted.startsWith "    drperf_gpu_device_" then emitted else
      s!"    drperf_gpu_instruction(__func__, \"{op}\", __LINE__, {scalarFlops + tensorWork}, {reads}, {writes}, {space}, {kind}, {addr}, {if unmodeled then 1 else 0});\n" ++
      latencyInstrumentation op args regs movVector wgmmaShape latencyPredicates ++ emitted

/- Standalone statement tests and generic callers retain fake-address guards.
   Whole-module translation supplies the statically inferred payload scope. -/
def translateStmt (stmt : String) : String :=
  translateStmtScoped stmt true

end Gxptx
