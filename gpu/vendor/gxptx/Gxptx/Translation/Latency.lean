/- Register dependencies for the optional, uncalibrated lane issue model.
   This describes timing assumptions, not PTX value semantics. -/
import Gxptx.Translation.Parsing
import Gxptx.Translation.C.Operands
namespace Gxptx

def latencyRegisters (regs : Std.HashSet String) (text : String) : List String :=
  let words := String.mk (text.toList.map (fun c =>
    if c.isAlphanum || c == '_' || c == '%' || c == '$' || c == '.' then c else ' '))
  ((words.splitOn " ").filter (fun r => !r.isEmpty && regs.contains r)).eraseDups

def latencyShadow (r : String) : String := "S_" ++ dest r

def latencyCall (cls : Nat) (src dst : List String) : String :=
  "    gx_latency(" ++ toString cls ++ ", {" ++
    String.intercalate ", " (src.eraseDups.map latencyShadow) ++ "}, {" ++
    String.intercalate ", " (dst.eraseDups.map (fun r => "&" ++ latencyShadow r)) ++ "});\n"

-- Braces have been flattened by Statement.lean. Recover the destination
-- extent from the same syntax/shape metadata used by value translation.
def latencyInstrumentation (op : String) (args : List String) (regs : Std.HashSet String)
    (movVector : Option Bool) (wgmmaShape : Option (Nat × Nat))
    (predicates : List String) : String :=
  let ts := op.splitOn "."
  let h := ts.headD ""
  let load := h == "ld" || h == "ldu" || h == "ldmatrix" || (h == "wmma" && ts.contains "load") ||
    (h == "tcgen05" && ts.contains "ld")
  let store := h == "st" || h == "stmatrix" || (h == "wmma" && ts.contains "store")
  let noDest := store || ["bra", "brx", "ret", "exit", "call", "cp", "red", "fence", "membar",
    "griddepcontrol", "nanosleep", "trap", "prefetch", "prefetchu", "discard", "setmaxnreg"].contains h ||
    ((h == "bar" || h == "barrier") && !ts.contains "red") ||
    (h == "wgmma" && !ts.contains "mma_async") ||
    (h == "mbarrier" && !(ts.contains "arrive" || ts.contains "try_wait" || ts.contains "test_wait")) ||
    (h == "tcgen05" && !ts.contains "ld" && !ts.contains "ld_red")
  let nd := if noDest then 0
    else if h == "wgmma" then wgmmaShape.map Prod.fst |>.getD 0
    else if h == "wmma" then
      if ts.contains "load" && ts.contains "bf16" && ts.contains "a" then (if ts.contains "m8n32k16" then 2 else if ts.contains "m32n8k16" then 8 else 4)
      else if ts.contains "load" && ts.contains "bf16" && ts.contains "b" then (if ts.contains "m8n32k16" then 8 else if ts.contains "m32n8k16" then 2 else 4)
      else 8
    else if h == "mma" then (if (ts.filter (fun t => ["f16", "f32", "s32"].contains t)).headD "f32" == "f16" then 2 else 4)
    else if h == "mov" && movVector == some false then args.length - 1
    else if h == "ldmatrix" then (if ts.contains "x4" then 4 else if ts.contains "x2" then 2 else 1)
    else if h == "ld" || h == "ldu" then (if ts.contains "v8" then 8 else if ts.contains "v4" then 4 else if ts.contains "v2" then 2 else 1)
    else if h == "tcgen05" && ts.contains "ld" then
      -- Register results precede the bracketed tensor address (and an
      -- optional immediate). ld.red also returns a reduction register.
      (args.takeWhile (fun a => !a.startsWith "[")).length
    else if h == "clusterlaunchcontrol" && ts.contains "v4" then 4
    else 1
  let dst := (args.take nd).flatMap (latencyRegisters regs)
  let src := predicates ++ (args.drop nd).flatMap (latencyRegisters regs) ++
    (if h == "wgmma" then dst else [])
  let shared := h == "ldmatrix" || h == "tcgen05" || ts.any (fun t => t == "shared" || t.startsWith "shared::")
  let global := !shared && !ts.any (fun t => ["param", "local", "const"].contains t || t.startsWith "param::")
  -- Classes: issue-only, ALU, global/generic, shared, MMA, SFU, atomic.
  let cls := if load then (if shared then 3 else if global then 2 else 1)
    else if h == "atom" then (if shared then 3 else 6)
    else if h == "mma" || (h == "wmma" && ts.contains "mma") || (h == "wgmma" && ts.contains "mma_async") then 4
    else if ["sin", "cos", "ex2", "lg2", "rcp", "rsqrt", "sqrt", "div"].contains h then 5
    else if noDest then 0 else 1
  latencyCall cls src dst

end Gxptx
