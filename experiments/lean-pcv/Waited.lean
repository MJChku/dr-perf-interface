import Lean

/-!
An algebra of observed region invocations and publication/checkpoint events.
This is an ordering specification, not a model of elapsed time or CPU blocking.
Region identities are invocation identities. Event identities include generation.
-/
namespace Waited

abbrev Event (n : Nat) := Fin n
abbrev Timeline (n : Nat) := Event n → Nat

/-- `members` is structural membership from the invocation trace, not inferred
    from overlapping wall-clock intervals. Equal names do not merge invocations. -/
structure Region (n : Nat) where
  invocation : Nat
  name : String
  members : List (Event n)
  deriving Repr

instance : Membership (Event n) (Region n) where
  mem R e := e ∈ R.members

instance : HasSubset (Region n) where
  Subset R S := ∀ e, e ∈ R → e ∈ S

theorem includes_refl (R : Region n) : R ⊆ R := fun _ h => h

theorem includes_trans {R S T : Region n} (h₁ : R ⊆ S) (h₂ : S ⊆ T) :
    R ⊆ T := fun _ h => h₂ _ (h₁ _ h)

/-- An atom constrains two event occurrences, never two entire regions. -/
structure Edge (n : Nat) where
  earlier : Event n
  later : Event n
  deriving Repr, DecidableEq

infix:50 " ≺ " => Edge.mk

def Holds (τ : Timeline n) (p : Edge n) : Prop := τ p.earlier < τ p.later

instance (τ : Timeline n) (p : Edge n) : Decidable (Holds τ p) :=
  inferInstanceAs (Decidable (τ p.earlier < τ p.later))

def Satisfies (τ : Timeline n) (Γ : List (Edge n)) : Prop :=
  ∀ p ∈ Γ, Holds τ p

instance (τ : Timeline n) (Γ : List (Edge n)) : Decidable (Satisfies τ Γ) :=
  inferInstanceAs (Decidable (∀ p ∈ Γ, τ p.earlier < τ p.later))

def Sat (Γ : List (Edge n)) : Prop := ∃ τ, Satisfies τ Γ

def Entails (Γ : List (Edge n)) (p : Edge n) : Prop :=
  ∀ τ, Satisfies τ Γ → Holds τ p

infix:45 " ⊨ " => Entails

/-- A waited marker has an actual checkpoint w. Retaining w is essential:
    two declarations for the same release at different checkpoints differ. -/
def WaitedAt (τ : Timeline n) (R : Region n) (e w : Event n) : Prop :=
  w ∈ R ∧ Holds τ (e ≺ w)

notation:45 R " ⊣[" τ ", " w "] " e => WaitedAt τ R e w

/-- e denotes the publication occurrence; this relation records its owner. -/
def Publishes (R : Region n) (e : Event n) : Prop := e ∈ R

infix:45 " ⊢ₚ " => Publishes

/-- Region abstraction preserves the release AND checkpoint witnesses. -/
def DependsWithin (τ : Timeline n) (consumer producer : Region n) (e w : Event n) : Prop :=
  (consumer ⊣[τ, w] e) ∧ (producer ⊢ₚ e)

theorem lift_wait {τ : Timeline n} {A B : Region n} {e w : Event n}
    (hBA : B ⊆ A) (h : B ⊣[τ, w] e) : A ⊣[τ, w] e :=
  ⟨hBA w h.1, h.2⟩

theorem lift_dependency {τ : Timeline n} {A B C D : Region n} {e w : Event n}
    (hBA : B ⊆ A) (hCD : C ⊆ D) (h : DependsWithin τ B C e w) :
    DependsWithin τ A D e w :=
  ⟨lift_wait hBA h.1, hCD e h.2⟩

theorem entails_member {Γ : List (Edge n)} {p : Edge n} (h : p ∈ Γ) : Γ ⊨ p :=
  fun _ hΓ => hΓ p h

theorem entails_trans {Γ : List (Edge n)} {a b c : Event n}
    (hab : Γ ⊨ (a ≺ b)) (hbc : Γ ⊨ (b ≺ c)) : Γ ⊨ (a ≺ c) :=
  fun τ hΓ => Nat.lt_trans (hab τ hΓ) (hbc τ hΓ)

/-- Reflexive closure of entailed strict order, useful for identical checkpoints. -/
def NoLater (Γ : List (Edge n)) (a b : Event n) : Prop := a = b ∨ Γ ⊨ (a ≺ b)

notation:45 a " ≼[" Γ "] " b => NoLater Γ a b

/-- A general coverage rule, including different releases when their dependency
    is established. No matching is inferred merely from being inside a parent. -/
theorem covers_via_order {Γ : List (Edge n)} {eB eA wA wB : Event n}
    (hRelease : eB ≼[Γ] eA) (hA : Γ ⊨ (eA ≺ wA)) (hPoint : wA ≼[Γ] wB) :
    Γ ⊨ (eB ≺ wB) := by
  have h : Γ ⊨ (eB ≺ wA) := by
    rcases hRelease with rfl | hRelease
    · exact hA
    · exact entails_trans hRelease hA
  rcases hPoint with rfl | hPoint
  · exact h
  · exact entails_trans h hPoint

/-- Exact criterion for deleting an assertion: preserve every model, not just
    the existence of some satisfying execution. -/
theorem drop_iff_entails (Γ : List (Edge n)) (p : Edge n) :
    (∀ τ, Satisfies τ (p :: Γ) ↔ Satisfies τ Γ) ↔ Γ ⊨ p := by
  constructor
  · intro h τ hΓ
    exact ((h τ).mpr hΓ) p (by simp)
  · intro h τ
    constructor
    · intro hΓ q hq
      exact hΓ q (by simp [hq])
    · intro hΓ q hq
      rcases List.mem_cons.mp hq with rfl | hq
      · exact h τ hΓ
      · exact hΓ q hq

theorem drop_preserves_sat {Γ : List (Edge n)} {p : Edge n} (h : Γ ⊨ p) :
    Sat (p :: Γ) ↔ Sat Γ := by
  constructor
  · rintro ⟨τ, hτ⟩
    exact ⟨τ, ((drop_iff_entails Γ p).mpr h τ).mp hτ⟩
  · rintro ⟨τ, hτ⟩
    exact ⟨τ, ((drop_iff_entails Γ p).mpr h τ).mpr hτ⟩

/-- A parent's earlier checkpoint covers a later child checkpoint for the SAME
    publication. The equality case is precisely witness-preserving lifting. -/
theorem parent_covers_child {Γ : List (Edge n)} {τ : Timeline n}
    {A B : Region n} {e wA wB : Event n}
    (_hBA : B ⊆ A) (hwB : wB ∈ B) (hΓ : Satisfies τ Γ)
    (hA : A ⊣[τ, wA] e) (horder : wA = wB ∨ Γ ⊨ (wA ≺ wB)) :
    B ⊣[τ, wB] e := by
  refine ⟨hwB, ?_⟩
  rcases horder with rfl | horder
  · exact hA.2
  · exact Nat.lt_trans hA.2 (horder τ hΓ)

/-- The usual outer waited AFTER the helper follows from the child, not vice versa. -/
theorem child_implies_later_parent {Γ : List (Edge n)} {e wB wA : Event n}
    (hB : Γ ⊨ (e ≺ wB)) (horder : Γ ⊨ (wB ≺ wA)) : Γ ⊨ (e ≺ wA) :=
  entails_trans hB horder

/-- A causal chain through B requires B's wait to precede B's publication. -/
theorem compose_dependencies {Γ : List (Edge n)} {eC wB eB wA : Event n}
    (hBC : Γ ⊨ (eC ≺ wB)) (hBridge : Γ ⊨ (wB ≺ eB))
    (hAB : Γ ⊨ (eB ≺ wA)) : Γ ⊨ (eC ≺ wA) :=
  entails_trans (entails_trans hBC hBridge) hAB

private theorem count_mono (xs : List α) (p q : α → Bool)
    (h : ∀ x ∈ xs, p x = true → q x = true) : xs.countP p ≤ xs.countP q := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have ht := ih (fun y hy => h y (by simp [hy]))
    have hh := h x (by simp)
    simp only [List.countP_cons]
    cases hp : p x <;> cases hq : q x <;> simp_all <;> omega

private theorem count_strict (xs : List α) (p q : α → Bool)
    (h : ∀ x ∈ xs, p x = true → q x = true)
    (w : ∃ x ∈ xs, p x = false ∧ q x = true) : xs.countP p < xs.countP q := by
  induction xs with
  | nil => simp at w
  | cons x xs ih =>
    have ht : ∀ y ∈ xs, p y = true → q y = true := fun y hy => h y (by simp [hy])
    have hm := count_mono xs p q ht
    have hh := h x (by simp)
    rcases w with ⟨y, hy, hp, hq⟩
    rcases List.mem_cons.mp hy with rfl | hy
    · simp only [List.countP_cons, hp, hq]
      simp only [Bool.false_eq_true, ↓reduceIte, Nat.add_zero]
      omega
    · have hi := ih ht ⟨y, hy, hp, hq⟩
      simp only [List.countP_cons]
      cases hx : p x <;> cases hy : q x <;> simp_all <;> omega

/-- Compress arbitrary natural timestamps without losing any strict ordering. -/
def rank (τ : Timeline n) (e : Event n) : Nat :=
  (List.finRange n).countP fun x => decide (τ x < τ e)

theorem rank_bounded (τ : Timeline n) (e : Event n) : rank τ e < n := by
  have h := count_strict (List.finRange n) (fun x => decide (τ x < τ e)) (fun _ => true)
    (fun _ _ _ => rfl) ⟨e, List.mem_finRange e, by simp, rfl⟩
  simpa [rank, List.countP_true] using h

theorem rank_lt_iff (τ : Timeline n) (a b : Event n) :
    rank τ a < rank τ b ↔ τ a < τ b := by
  constructor
  · intro h
    by_cases hab : τ a < τ b
    · exact hab
    · have hba : τ b ≤ τ a := by omega
      have hm := count_mono (List.finRange n) (fun x => decide (τ x < τ b))
        (fun x => decide (τ x < τ a)) (by
          intro x _ hx
          apply decide_eq_true
          have hx' := of_decide_eq_true hx
          omega)
      change rank τ b ≤ rank τ a at hm
      omega
  · intro h
    apply count_strict
    · intro x _ hx
      apply decide_eq_true
      exact Nat.lt_trans (of_decide_eq_true hx) h
    · exact ⟨a, List.mem_finRange a, by simp, by simpa using h⟩

def compress (τ : Timeline n) (e : Event n) : Fin n := ⟨rank τ e, rank_bounded τ e⟩

theorem holds_compress (τ : Timeline n) (p : Edge n) :
    Holds (fun e => (compress τ e).val) p ↔ Holds τ p :=
  rank_lt_iff τ p.earlier p.later

theorem satisfies_compress (τ : Timeline n) (Γ : List (Edge n)) :
    Satisfies (fun e => (compress τ e).val) Γ ↔ Satisfies τ Γ := by
  simp only [Satisfies, holds_compress]

/-- Exhaustive finite schedules: a small reference decision procedure, not a
    scalable production algorithm. n events require at most n distinct ranks. -/
def assignments (m k : Nat) : List (Fin m → Fin k) :=
  match m with
  | 0 => [Fin.elim0]
  | m + 1 => (List.finRange k).flatMap fun head =>
      (assignments m k).map fun tail => Fin.cases head tail

theorem mem_assignments (f : Fin m → Fin k) : f ∈ assignments m k := by
  induction m with
  | zero =>
    have hf : f = Fin.elim0 := by funext i; exact Fin.elim0 i
    simp [assignments, hf]
  | succ m ih =>
    simp only [assignments, List.mem_flatMap, List.mem_map]
    refine ⟨f 0, List.mem_finRange _, (fun i => f i.succ), ih _, ?_⟩
    funext i
    exact Fin.cases rfl (fun _ => rfl) i

def sat? (Γ : List (Edge n)) : Bool :=
  (assignments n n).any fun τ => decide (Satisfies (fun e => (τ e).val) Γ)

def entails? (Γ : List (Edge n)) (p : Edge n) : Bool :=
  (assignments n n).all fun τ =>
    decide (Satisfies (fun e => (τ e).val) Γ → Holds (fun e => (τ e).val) p)

def canDrop? (Γ : List (Edge n)) (p : Edge n) : Bool := sat? Γ && entails? Γ p

/-- Validate the actual recorded timeline separately from asking whether some
    hypothetical timeline could satisfy the same claims. -/
def accepts? (τ : Timeline n) (Γ : List (Edge n)) : Bool := decide (Satisfies τ Γ)

theorem accepts?_correct (τ : Timeline n) (Γ : List (Edge n)) :
    accepts? τ Γ = true ↔ Satisfies τ Γ := by simp [accepts?]

theorem sat?_sound {Γ : List (Edge n)} (h : sat? Γ = true) : Sat Γ := by
  obtain ⟨τ, _, hτ⟩ := List.any_eq_true.mp h
  exact ⟨fun e => (τ e).val, of_decide_eq_true hτ⟩

theorem sat?_correct (Γ : List (Edge n)) : sat? Γ = true ↔ Sat Γ := by
  constructor
  · exact sat?_sound
  · rintro ⟨τ, hτ⟩
    apply List.any_eq_true.mpr
    exact ⟨compress τ, mem_assignments _,
      decide_eq_true ((satisfies_compress τ Γ).mpr hτ)⟩

theorem entails?_correct (Γ : List (Edge n)) (p : Edge n) :
    entails? Γ p = true ↔ Γ ⊨ p := by
  constructor
  · intro h τ hτ
    have hc := (List.all_eq_true.mp h) (compress τ) (mem_assignments _)
    have hp := (of_decide_eq_true hc) ((satisfies_compress τ Γ).mpr hτ)
    exact (holds_compress τ p).mp hp
  · intro h
    apply List.all_eq_true.mpr
    intro τ _
    exact decide_eq_true (h (fun e => (τ e).val))

theorem canDrop?_correct (Γ : List (Edge n)) (p : Edge n) :
    canDrop? Γ p = true ↔ Sat Γ ∧ Γ ⊨ p := by
  simp only [canDrop?, Bool.and_eq_true, sat?_correct, entails?_correct]

theorem canDrop_preserves_models {Γ : List (Edge n)} {p : Edge n}
    (h : canDrop? Γ p = true) : ∀ τ, Satisfies τ (p :: Γ) ↔ Satisfies τ Γ :=
  (drop_iff_entails Γ p).mpr ((canDrop?_correct Γ p).mp h).2

/-- An unknown publisher cannot be identified merely by choosing a convenient
    satisfying assignment. This conservative rule covers EVERY supplied candidate. -/
def coversUnknown? (Γ : List (Edge n)) (candidates : List (Event n)) (w : Event n) : Bool :=
  sat? Γ && !candidates.isEmpty && candidates.all (fun e => entails? Γ (e ≺ w))

theorem coversUnknown_sound {Γ : List (Edge n)} {candidates : List (Event n)} {w : Event n}
    (h : coversUnknown? Γ candidates w = true) : ∀ e ∈ candidates, Γ ⊨ (e ≺ w) := by
  have hall := (Bool.and_eq_true _ _ ▸ h).2
  intro e he
  exact (entails?_correct Γ (e ≺ w)).mp (List.all_eq_true.mp hall e he)

end Waited
