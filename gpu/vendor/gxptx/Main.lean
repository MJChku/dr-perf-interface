/-
  gxptx (Lean) — driver for the verification-accounted PTX→C translator.
  Reads a .ptx module, emits a compilable C translation unit:
  PRELUDE + one C function per PTX kernel/.func, with operator selection
  routed through the theorem-backed modeled core; trusted and known-gap forms
  are reported separately by `lake exe gxcheck`.
-/
import Gxptx.Translation
-- Importing the verification aggregate makes every production theorem and
-- total certificate elaborate as a direct dependency of the translator CLI.
import Gxptx.Verification

/- PTX definition: accepted instruction text is dispatched by Translation.Statement; unsupported or incomplete forms remain explicit traps or known gaps. -/
/- Lean modeling: importing the Translation and Verification aggregates makes executable code and all certificates direct, reviewable dependencies. -/
def main (args : List String) : IO Unit := do
  match args with
  | inPath :: rest => do
      let src ← IO.FS.readFile inPath
      -- PRELUDE path: env override, else source tree.
      let envp ← IO.getEnv "GXPTX_PRELUDE"
      let preludePaths := (envp.toList) ++ [
        "tools/gxptx/lean/Gxptx/prelude.h", "Gxptx/prelude.h" ]
      let mut prelude := ""
      for p in preludePaths do
        if prelude.isEmpty then
          if ← System.FilePath.pathExists p then
            prelude := (← IO.FS.readFile p)
      -- GXPTX_SPLIT=N → write <out>.0.cpp … <out>.(N-1).cpp for parallel compile
      let splitEnv ← IO.getEnv "GXPTX_SPLIT"
      let nsplit := (splitEnv.bind (·.toNat?)).getD 1
      match rest with
      | outPath :: _ =>
        if nsplit > 1 then
          let chunks := Gxptx.translateModuleChunks prelude src nsplit
          for i in [0:nsplit] do
            IO.FS.writeFile s!"{outPath}.{i}.cpp" chunks[i]!
        else
          IO.FS.writeFile outPath (Gxptx.translateModule prelude src)
      | [] => IO.print (Gxptx.translateModule prelude src)
  | [] => IO.eprintln "usage: gxptx <in.ptx> [out.c]"
