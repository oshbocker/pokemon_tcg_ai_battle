# Observation / Action Encoding Contract

The spec the model implements against. It is the prose companion to
[`src/ptcg_battle/encoding.py`](../src/ptcg_battle/encoding.py); the round-trip
invariants are enforced in [`tests/test_encoding.py`](../tests/test_encoding.py).
If you change one, change all three (the `prepare` gate checks they stay in
sync).

> **Status:** Phase 1 (P1.3/P1.4). Encoding is implemented and tested; the model
> that consumes it (Phase 2) is not built yet — section
> [§5 Model interface](#5-model-interface-pointer--value-heads) is the agreed
> shape, not yet code.

## 0. Principles (why it looks like this)

- **Read the raw JSON dict, never the dataclass.** Phase 0 measured that
  `to_observation_class()` ~halves engine throughput (recursive Python
  `to_dataclass`). The encoder indexes `obs["..."]` directly. See
  [`PHASE0_THROUGHPUT.md`](../rl_research/PHASE0_THROUGHPUT.md) finding #2.
- **Low-level, entity-based; let the model learn the cards** (Lessons 3–4). Card
  knowledge lives in a learned **card-ID embedding** over all 1267 cards — we do
  *not* hand-encode "Crustle walls ex damage". Static `card_table` metadata
  (type, weakness, …) is available but deliberately **not** baked into the
  default features; the ID embedding is expected to learn it from outcomes.
- **Pointer action head, no action mask.** The engine hands us a variable-length
  list of **only-legal** options every decision (`select.option`). We emit one
  **candidate token per option**, index-aligned, and the head scores them. The
  candidate set *is* the legality mask.
- **POMDP.** We encode the **observation**, not ground truth: own hand is full,
  the opponent's is a count, deck order and face-down prizes are hidden. The
  critic sees exactly what the policy sees.

## 1. When the encoder runs

`encode_observation(obs)` is called at every decision point — i.e. whenever
`obs["select"] is not None`. The **initial deck selection** (`select is None`) is
*not* encoded: the agent just returns the 60-card deck list (`agent/deck.csv`),
which is fixed and outside the action space. Passing a deck-selection obs to the
encoder raises.

Everything is from the **acting player's point of view** (`my_index =
current.yourIndex`); "own"/"opp" below are relative to that seat, so the same
network plays either side (seat side-swap in eval is just data, not a code path).

## 2. Vocabularies (embedding tables)

Index `0` is reserved as PAD / "none" / face-down in **every** table.

| Table | Size constant | Range | Source |
|---|---|---|---|
| Card ID | `CARD_VOCAB = 1268` | ids `1..1267` | `all_card_data()` — dense |
| Attack ID | `ATTACK_VOCAB = 1557` | ids `1..1556` | `all_attack()` — dense |
| Energy type | `N_ENERGY_TYPES = 12` | `0..11` | `EnergyType` |
| Area type | `N_AREA = 13` | `1..12` (0=none) | `AreaType` |
| Option type | `N_OPTION_TYPE = 17` | `0..16` | `OptionType` |
| Select context | `N_SELECT_CONTEXT = 49` | `0..48` | `SelectContext` |
| Entity role | `N_ROLE = 8` | see §3 | encoder-defined |
| Special condition | `N_SPECIAL_COND = 6` | `0=none, 1..5` | `SpecialConditionType + 1` |

The engine may append new cards/attacks/contexts mid-competition (the enums say
so). `test_declared_vocab_bounds_live_engine` asserts the live engine still fits
these sizes, so a drop trips a test rather than silently corrupting an embedding.
If it trips: bump the constant, grow the embedding row count, keep old rows.

## 3. Entity tokens (the board as a set)

`encode_observation` emits a variable-length list of entity tokens, each with:

- `entity_role: int64` — one of `PAD=0, OWN_ACTIVE=1, OWN_BENCH=2, OPP_ACTIVE=3,
  OPP_BENCH=4, OWN_HAND=5, GLOBAL=6, STADIUM=7`.
- `entity_card: int64` — card-ID embedding index (0 if none/face-down).
- `entity_feat: float32[15]` — normalised numerics (layout below).
- `entity_energy: float32[12]` — per-`EnergyType` attached-energy count ÷ 6.

Emission order (deterministic):

1. **Own active** (skipped if the slot is empty/face-down).
2. **Own bench** (each Pokémon, in engine order).
3. **Opp active**, then **opp bench**.
4. **Own hand** cards (`role=OWN_HAND`; card-ID only, no Pokémon stats).
   The opponent's hand is hidden → only its **count** appears, in `global_feat`.
5. **Stadium** token (`role=STADIUM`) if a stadium is in play (shared zone).
6. **Global** token (`role=GLOBAL`, always last) carrying `global_feat`.

A face-down opponent active contributes **no token** (we can't see its identity),
but it is still reflected in the prize/bench counts of `global_feat`. The
opponent active's true id is recoverable only via the search API's
`opponent_active` prediction, which the encoder does not use.

`entity_feat[15]` (all ≈[0,1], status flags only meaningful on the active):

```
0 hp/300            5 n_tools/2          10 poisoned
1 maxHp/300         6 n_preEvolution/2   11 burned
2 hp/maxHp          7 appearThisTurn     12 asleep
3 n_energies/6      8 is_active          13 paralyzed
4 n_energyCards/6   9 is_own             14 confused
```

`global_feat[18]` (board summary, "my" POV, ≈[0,1]; opp hand is count-only here):

```
0 turn/40            6 my_hand/12          12 energyAttached
1 turnAction/20      7 opp_hand/12         13 retreated
2 my_prizes/6        8 my_bench/5          14 i_go_first (firstPlayer==me)
3 opp_prizes/6       9 opp_bench/5         15 stadium_present
4 my_deck/60        10 supporterPlayed     16 my_discard/60
5 opp_deck/60       11 stadiumPlayed       17 opp_discard/60
```

## 4. Option / candidate tokens (the action space)

One candidate token **per legal option**, index-aligned with
`obs["select"]["option"]`. **Candidate `i` is the pointer-head logit for choosing
option `i`** — this 1:1, total, positional mapping is the contract's headline
invariant (`test_one_candidate_token_per_legal_option`). Per candidate:

- `opt_type: int64` — `OptionType` verbatim.
- `opt_area`, `opt_inplay_area: int64` — `AreaType` of the referenced card and of
  the in-play target Pokémon (0 if absent).
- `opt_card: int64` — **resolved** card-ID of what the option refers to (see
  resolution table), 0 if none. This is the key link from "option" to "which
  card" so the head can reason about identity.
- `opt_attack: int64` — attack-ID embedding index (ATTACK only; else 0).
- `opt_special: int64` — `SpecialConditionType + 1` (SPECIAL_CONDITION only).
- `opt_feat: float32[6]` — `[number/20, count/6, has_card, has_toolIndex,
  has_energyIndex, targets_opponent]`.
- `opt_target: int64` — **index of the entity token this option acts on**, so the
  pointer can attend option→target (e.g. ATTACH energy → the receiving Pokémon).
  Never −1 in the output: options with no board referent (END, YES/NO, NUMBER,
  RETREAT) fall back to the **global token index**, guaranteeing a valid key.
- `opt_rank: int64` — the option's **index in the engine's enumeration**, clamped to
  `N_OPTION_RANK−1` (= 63). The engine sorts `select.option` strong→weak — returning
  option 0 ("B1") beats random ~88–90% (Kaggle #713608) — and a pointer head is
  permutation-invariant, so without this it discards the engine's prior. **Ablatable:
  the encoder always emits it; the model gates it with a `use_option_rank` flag** (a
  learned positional embedding added to candidate tokens), so we can A/B whether it
  helps. See `rl_research/PHASE1_RESEARCH.md`.

Card resolution per `OptionType` (raw-dict analogue of `main.py:get_card`):

| OptionType | `opt_card` resolves to | `opt_target` |
|---|---|---|
| `PLAY` (7) | own hand[`index`] | global (the card is in hand) |
| `ATTACH` (8) | own `area`[`index`] (hand card) | `inPlay(Area,Index)` Pokémon |
| `EVOLVE` (9) | own `area`[`index`] (evolution card) | `inPlay(Area,Index)` Pokémon |
| `CARD` (3) | `(area,index,playerIndex)` card | that card's token if in play |
| `ABILITY` (10) / `DISCARD` (11) | `(area,index)` card in play | that Pokémon token |
| `TOOL_CARD` (4) | the Pokémon's `tools[toolIndex]` | the host Pokémon token |
| `ENERGY_CARD` (5) / `ENERGY` (6) | the Pokémon's `energyCards[energyIndex]` | the host Pokémon token |
| `ATTACK` (13) | — (uses `opt_attack`) | own active token |
| `RETREAT` (12) / `END` (14) / `YES` (1) / `NO` (2) | — | global token |
| `NUMBER` (0) | — (`number` in `opt_feat`) | global token |
| `SPECIAL_CONDITION` (16) | — (`opt_special`) | global token |
| `SKILL` (15) | — (`cardId`/`serial` ordering; treated as no-card) | global token |

`playerIndex` defaults to the acting player when an option omits it (PLAY/ATTACH/
EVOLVE are always our own cards). `CARD` options carry an explicit `playerIndex`
(e.g. choosing an opponent's Pokémon to damage), and `targets_opponent` flags it.

## 5. Model interface (pointer + value heads)

*(Phase 2 — agreed shape, not yet implemented.)*

**Trunk.** Embed every token (entity + candidate) into `d_model` by summing its
categorical embeddings and a linear projection of its numeric features:

- entity: `role_emb + card_emb + Linear(entity_feat) + Linear(entity_energy)`,
  with `global_feat` projected onto the GLOBAL token.
- candidate: `opt_type_emb + area_emb + inplay_area_emb + card_emb + attack_emb +
  special_emb + Linear(opt_feat)`, **plus the embedding of its `opt_target`
  entity token** (the option→target link), **and — when `use_option_rank` is on —
  a learned `opt_rank` positional embedding** (the engine's best→worst prior).

Run all tokens through a shared transformer (Phase 2 starts ~6–12 blocks,
`d_model` 256–512; sizing gated by the P0.3 inference probe). Include a couple of
learned **summary/scratch tokens** (the winner's global-workspace trick).

**Pointer actor head.** Build a **query** from the pooled trunk output + an
embedding of the `SelectContext`. Score each candidate token `c_i` by scaled
dot-product `q · k_i / √d` over **only the current candidate set**; softmax →
policy. Legality is free (only legal options are tokens), so **no action mask**.

**Multi-pick** (`max_count > 1`, e.g. `DISCARD`, damage-counter spreads): the
engine wants a set of distinct indices with `min_count ≤ |set| ≤ max_count`.
Default plan = **autoregressive pointer**: pick one candidate, mask the chosen
index, re-score, repeat until an emitted "stop" (allowed once `min_count` met) or
`max_count`. The encoder supplies the bounds (`min_count`, `max_count`) and the
candidate set; the *already-picked* mask is runtime state the model threads
through the step, **not** part of the obs encoding. (Alternative considered:
independent Bernoulli per candidate with a count constraint — rejected for now
because it can't model "pick the best 2 *together*". Revisit in P2.3.)

**Critic head.** A value token (or pooled summary) → `tanh` → value in `[-1,1]`,
trained on the terminal ±1 / 0-draw reward. POMDP ⇒ the critic sees only the
observation (same as the policy).

## 6. Invariants (enforced in tests)

1. `n_options == len(select.option)`; `opt_type == [o.type for o in option]`
   (1:1 positional mapping — the headline invariant).
2. `0 ≤ opt_target < n_entities` for every candidate (valid attention key).
3. All IDs within their vocab; `opt_special < N_SPECIAL_COND`;
   `context < N_SELECT_CONTEXT`.
4. Features finite and in range (no NaN/Inf; a unit-slip ceiling).
5. Deterministic: same dict → identical arrays.
6. Independent cross-check vs the dataclass parse: every `PLAY` candidate's
   `opt_card` is a card in our parsed hand; every `ATTACK` candidate's
   `opt_attack` equals the engine `attackId`.
7. `0 ≤ min_count ≤ max_count ≤ n_options`; a multi-pick decision is present in
   the fixture.
8. Declared vocab sizes still bound the live engine.
9. `opt_rank == clamp(arange(n_options), N_OPTION_RANK−1)` (the engine-order prior).
