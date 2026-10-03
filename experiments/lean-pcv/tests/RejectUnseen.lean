import Examples
import Discover
open Pcv Examples

attribute [local simp] x light linear

-- All discovery samples have x <= 11, so they miss the extra operation.
def lateChange : Pcv.Expr := .iteLt x (.lit 1000) linear (.seq light linear)
attribute [local simp] lateChange

#guard match inferInterface lateChange [x] with
  | .ok f => f.constant == 2 && f.terms.map Prod.fst == [3]
  | .error _ => false

-- This must fail at the proof stage, despite the exact sample fit.
perf_discover falseFit : lateChange using [x]
