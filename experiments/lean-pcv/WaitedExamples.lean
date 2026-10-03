import Waited

namespace Waited.Examples
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

/- The event numbers are identities, NOT timestamps. All possible timestamp
   assignments are considered by sat?, entails?, and canDrop?. -/

def R_A : Region 3 := ⟨0, "A", [0, 2]⟩
def R_B : Region 3 := ⟨1, "B", [0]⟩
def R_C : Region 3 := ⟨2, "C", [1]⟩
def R_D : Region 3 := ⟨3, "D", [1]⟩

theorem B_in_A : R_B ⊆ R_A := by
  change ∀ e : Fin 3, e ∈ ([0] : List (Fin 3)) → e ∈ ([0, 2] : List (Fin 3))
  decide

theorem C_in_D : R_C ⊆ R_D := by
  change ∀ e : Fin 3, e ∈ ([1] : List (Fin 3)) → e ∈ ([1] : List (Fin 3))
  decide

/-- Display-level hiding preserves the very same checkpoint (0), NOT A's later
    checkpoint (2). No new temporal dependency is introduced. -/
theorem hide_helpers (τ : Timeline 3) (h : DependsWithin τ R_B R_C 1 0) :
    DependsWithin τ R_A R_D 1 0 := lift_dependency B_in_A C_in_D h

/-- Counterexample: B returns, C publishes, THEN A executes waited. A is correct
    while a claim that B had already waited for the same release is false. -/
theorem parent_wait_does_not_cover_earlier_child :
    (R_B ⊆ R_A) ∧ (R_A ⊣[(fun e => e.val), 2] 1) ∧
      ¬ (R_B ⊣[(fun e => e.val), 0] 1) := by
  refine ⟨B_in_A, ?_, ?_⟩
  · change (2 : Fin 3) ∈ ([0, 2] : List (Fin 3)) ∧ (1 : Nat) < 2
    decide
  · change ¬ ((0 : Fin 3) ∈ ([0] : List (Fin 3)) ∧ (1 : Nat) < 0)
    decide

def parentAfter : List (Edge 3) := [0 ≺ 1, 1 ≺ 2]
theorem parentAfter_sat : sat? parentAfter = true := by decide
theorem childWouldContradict : sat? ((1 ≺ 0) :: parentAfter) = false := by decide
theorem cannotDropEarlierChild : canDrop? parentAfter (1 ≺ 0) = false := by decide

/-- A's checkpoint (1) precedes B's (2); the parent already enforces e=0. -/
def parentBefore : List (Edge 3) := [0 ≺ 1, 1 ≺ 2]
theorem canDropLaterChild : canDrop? parentBefore (0 ≺ 2) = true := by decide
theorem droppingLaterChildPreservesAllTimelines :
    ∀ τ, Satisfies τ ((0 ≺ 2) :: parentBefore) ↔ Satisfies τ parentBefore :=
  canDrop_preserves_models canDropLaterChild

theorem sameCheckpoint : canDrop? ([(0 : Event 2) ≺ 1]) (0 ≺ 1) = true := by decide

/-- Merely finding a schedule satisfying both declarations proves no redundancy. -/
def parentOnly : List (Edge 3) := [0 ≺ 1]
theorem compatible : sat? ((0 ≺ 2) :: parentOnly) = true := by decide
theorem compatibleButNotRedundant : canDrop? parentOnly (0 ≺ 2) = false := by decide

/-- SAT of declarations alone is NOT validation of this recorded execution. -/
theorem wrongTimeline : accepts? (fun e : Event 3 => 2 - e.val) parentOnly = false := by decide

/-- Two uses of the same helper name have different invocation identities.
    Neither their scopes nor their event generations are merged. -/
def R_B₁ : Region 4 := ⟨10, "sync", [1]⟩
def R_B₂ : Region 4 := ⟨11, "sync", [3]⟩
def separateCalls : List (Edge 4) := [0 ≺ 1, 2 ≺ 3]
theorem distinctInvocations : R_B₁.invocation ≠ R_B₂.invocation := by decide
theorem noCrossCallerInference : entails? separateCalls (0 ≺ 3) = false := by decide

/-- Knowing A waits for one release does not identify B's unknown release. -/
def partialKnowledge : List (Edge 4) := [0 ≺ 1, 1 ≺ 3]
theorem knownReleaseCovered : coversUnknown? partialKnowledge [0] 3 = true := by decide
theorem unknownReleaseNotCovered : coversUnknown? partialKnowledge [0, 2] 3 = false := by decide
theorem noCandidatesNotCovered : coversUnknown? partialKnowledge [] 3 = false := by decide

/-- B publishes to A before B subsequently waits for C: regional arrows alone
    do NOT imply that A waits for C. -/
def publishedEarly : List (Edge 4) := [0 ≺ 1, 2 ≺ 3, 0 ≺ 3]
theorem noBlindTransitivity : entails? publishedEarly (2 ≺ 1) = false := by decide
def connectedChain : List (Edge 4) := [2 ≺ 3, 3 ≺ 0, 0 ≺ 1]
theorem connectedTransitivity : canDrop? connectedChain (2 ≺ 1) = true := by decide

def cycle : List (Edge 2) := [0 ≺ 1, 1 ≺ 0]
theorem rejectsCycle : sat? cycle = false := by decide
/-- Inconsistent premises must not give the simplifier permission to erase facts. -/
theorem rejectsVacuousCoverage : canDrop? cycle (0 ≺ 1) = false := by decide

#eval ("SAT parent; child can still be wrong", sat? parentAfter, canDrop? parentAfter (1 ≺ 0))
#eval ("Earlier parent checkpoint covers later child", canDrop? parentBefore (0 ≺ 2))
#eval ("Compatible does not mean redundant", sat? ((0 ≺ 2) :: parentOnly), canDrop? parentOnly (0 ≺ 2))
#eval ("Unknown release remains uncovered", coversUnknown? partialKnowledge [0, 2] 3)
#eval ("Shared helper does not mix callers", entails? separateCalls (0 ≺ 3))
#eval ("Cycles rejected", sat? cycle)

#print axioms lift_dependency
#print axioms covers_via_order
#print axioms drop_iff_entails
#print axioms parent_covers_child
#print axioms compose_dependencies
#print axioms rank_lt_iff
#print axioms sat?_correct
#print axioms entails?_correct
#print axioms canDrop_preserves_models
#print axioms coversUnknown_sound
#print axioms parent_wait_does_not_cover_earlier_child
#print axioms cannotDropEarlierChild

end Waited.Examples
