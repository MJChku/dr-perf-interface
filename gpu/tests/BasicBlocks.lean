import Gxptx.Translation.Body
open Gxptx

def counter (op : String) : String :=
  s!"drperf_gpu_instruction(__func__, \"{op}\", __LINE__, 0, 0, 0, 0, 0, 0, 0);"
def inBlock (id : Nat) (op : String) : String :=
  (counter op).replace "__func__" s!"\"kernel:bb{id}\""

-- Predication changes executed cost, not static block membership.
#guard profileBasicBlocks ["if (p) { " ++ counter "ld.global.f32" ++ " }",
    "if (!p) { " ++ counter "ld.global.f32" ++ " }", counter "ret"] "kernel" ==
  ["if (p) { " ++ inBlock 0 "ld.global.f32" ++ " }",
   "if (!p) { " ++ inBlock 0 "ld.global.f32" ++ " }", inBlock 0 "ret"]

-- Conditional transfer ends the block; target labels start it, including loops.
#guard profileBasicBlocks [counter "bra", counter "add.f32", "L_loop: ;",
    counter "mul.f32", counter "bra.uni", "L_done: ;", counter "ret"] "kernel" ==
  [inBlock 0 "bra", inBlock 1 "add.f32", "L_loop: ;",
   inBlock 2 "mul.f32", inBlock 2 "bra.uni", "L_done: ;", inBlock 3 "ret"]

-- Calls/returns delimit continuations; no empty blocks for consecutive labels.
#guard profileBasicBlocks [counter "call.uni", "L_a: ;", "L_b: ;",
    counter "ret", counter "exit"] "kernel" ==
  [inBlock 0 "call.uni", "L_a: ;", "L_b: ;", inBlock 1 "ret", inBlock 2 "exit"]

-- Markers have no counter but delimit the region whose counts are fitted.
#guard profileBasicBlocks [counter "add.f32", "drperf_gpu_device_end();",
    counter "mul.f32"] "kernel" ==
  [inBlock 0 "add.f32", "drperf_gpu_device_end();", inBlock 1 "mul.f32"]
