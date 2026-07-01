# Competitive TCG meta research + Kaggle-pool reconciliation — 2026-07-01

Deep-research sweep (r/pkmntcg + LimitlessTCG/RK9/NAIC results, adversarially
verified: 46 claims extracted → 14 confirmed, 11 refuted) cross-checked against
(a) our measured Kaggle ladder meta (226-replay snapshot, `analyze_meta_replays.py`)
and (b) our competition card pool (`data/EN_Card_Data.csv`). Purpose: pick 2-3
strategically-distinct deck seeds for the coevolution/mutation roster, and verify
our Archaludon + Alakazam seeds start "from strength."

## The load-bearing finding: real-world strength ≠ Kaggle strength, and our pool is CURATED

The cabt engine's card pool is **not full real-world Standard**. Every meta
archetype's *Pokémon lines* are present, but many **universal Standard staple
trainers are ABSENT**:

| Absent from our pool | Present substitute in pool |
|---|---|
| Iono (hand disruption) | Judge |
| Arven (item tutor) | — (Ultra Ball for search) |
| Professor's Research (draw) | Dawn, Hilda, Cheren, Fennel, Lillie's Determination |
| Nest Ball, Counter Catcher, Super Rod, Earthen Vessel | Ultra Ball, Night Stretcher, Boss's Orders (gust IS present) |
| Darkness Energy (special) | Basic {D} Energy |

**Consequence:** you cannot copy an optimized human decklist 1:1 — the Pokémon
core transfers, but the **trainer/energy shell must be rebuilt from our pool**.
This is *good* for the coevolution thesis: the optimal in-pool shell is unknown
and must be *discovered* (search), not copied. It also means our own two decks'
Kaggle strength is partly an artifact of which counters exist here.

Two concrete reconciliation wins from the pool curation:
- **Our Alakazam is likely BETTER in Kaggle than in real Standard.** Its verified
  real-world weakness is inconsistency under **Iono** disruption — and *Iono is not
  in our pool*. A key predator is simply absent.
- **Archaludon's real-world decline may not transfer** either (the counters that
  pushed it out of real top-8 may be absent) — but see the yellow flag below.

## Reconciliation table (real-world tier vs our Kaggle meta)

Kaggle numbers = distinct-pilot % and win-rate from the 226-replay snapshot.
Real-world = verified competitive standing (mid-2026, Mega Evolution era).

| Archetype | Game plan | Kaggle pilots% / WR | Real-world standing (verified) | Pool legal? |
|---|---|---|---|---|
| **Alakazam combo** (our B) | Psychic hand-scaling OHKO | **32% / 51%** (#1) | Tier-1.5: 6th (~4.75%), 4 Reg T8 (1 win), **NAIC 2026 2nd** | ✅ (MEG line) |
| **Archaludon ex** (our A) | Metal big-attacker/control | **13% / 49%** (#2) | **Declined / not tier-1-2**; sub-50% 2026 regionals | ✅ |
| **Marnie's Grimmsnarl ex** | Spread + disruption control | **7% / 57%** | **Sustained tier-1**: 39 Reg T8/7 wins, 10 Intl T8/2 wins | ✅ (DRI+TWM) |
| **Dragapult ex** | Tempo / bench-spread | **4% / 47%** | **Format-defining**: ~4 of 8 NAIC 2026 top-8 | ✅ (TWM+SFA) |
| Chandelure | Fire spread | 3% / 60% | not highlighted | ✅ |
| Cynthia's Garchomp ex | Fighting tempo | 4% / 57% | — | ✅ |
| Mega Starmie / Mega Lucario | Water tempo / Fighting | 10% / 43-47% | — | ✅ |
| Clefairy / single-prize | Low-variance single-prize | ~0 (no data) | **won NAIC 2026 outright** | ✅ ("Clefairy" POR; NOT "Lillie's Clefairy ex") |

## Verdict on our two seeds

- **Alakazam (B): keep, strong seed.** Real tier-1.5 AND Kaggle #1, and its main
  real predator (Iono) is absent here. Highest-confidence seed.
- **Archaludon (A): keep, but flag headroom.** Empirically our best Kaggle agent
  (1047, #2 pilots) — Kaggle strength is what the ladder rewards, so do NOT drop it.
  But real-world data says the *archetype* has a **lower power ceiling** than
  Dragapult/Grimmsnarl. Read: Archaludon may be a local-meta artifact; in the
  coevolution let stronger archetypes potentially overtake it rather than
  over-investing. The real-world tier is a proxy for *archetype headroom*.

## Recommended roster additions (distinct game plans, proven, legal)

Current plans: Metal control (Archaludon) + Psychic combo (Alakazam). Add:

1. **Marnie's Grimmsnarl ex — HIGH CONFIDENCE.** Strong in *both* worlds
   (real tier-1; Kaggle 57% WR, and it's the deck that beat our Archaludon).
   Distinct plan: damage-*spread control* (Froslass Freezing Shroud + Munkidori
   Adrena-Brain relocate counters; Grimmsnarl ex Shadow Bullet 180+30). Core lines
   legal: Marnie's Impidimp/Morgrem/Grimmsnarl ex (DRI), Munkidori/Snorunt/Froslass
   (TWM), Basic {D} Energy. Rebuild trainer shell (no Iono/Arven → Judge/Dawn/Hilda).
2. **Dragapult ex — HIGH CONFIDENCE, likely Kaggle-underexploited.**
   Format-defining in real play but only 4% of Kaggle pilots → probable *alpha*
   (a strong archetype the ladder hasn't saturated). Distinct plan: tempo +
   Phantom Dive bench-spread, often + Dusknoir (Cursed Blast) for control finish.
   Core lines legal: Dreepy/Drakloak/Dragapult ex (TWM), Duskull/Dusclops/Dusknoir
   (SFA), Budew, Rare Candy, Buddy-Buddy Poffin.
3. **Third pick = a strategic fork (pick one):**
   - *Clefairy / single-prize aggro* — real NAIC-winner, adds the **fast low-variance
     tempo** plan our slow decks lack; robust (few dead draws). "Clefairy" (POR) legal.
   - *Chandelure* — Kaggle-proven (60% WR), fire spread; adds pressure without new risk.
   - *Crustle stall/deckout* — the **orthogonal** plan; our agents already fold to
     stall (Alakazam 0-4 vs Hop's Snorlax), so a stall seed hardens the whole league.
     Legal, but not a real-world "winner."

## Caveats (from the verified research)

- Adversarial verification **killed** these — do NOT trust: Dragapult "49.22% share",
  Archaludon "13 Regional Top-8s / tier-1", Archaludon's "Raging Hammer 80+10/counter"
  attack + matchup spread, Grimmsnarl "0.35% low-share", the "10x MEE#8 / 3x Archaludon"
  list. Archetype-level standings survived; specific numbers/attack text did not.
- JustInBasil tier snapshot is dated **2025-07-20** (~1yr stale); 2026 LimitlessTCG/NAIC
  results confirm the *direction* (Archaludon down, Grimmsnarl sustained).
- Exact decklists were only verified for Alakazam (core line) + Grimmsnarl; Dragapult
  and Clefairy counts must be pulled fresh from LimitlessTCG before seeding.
- Every archetype's key line leans on a recent set (MEG for Alakazam, DRI for
  Grimmsnarl, TWM/SFA for Dragapult) — all in-pool, but single-set dependencies.

## Seeds built + validated (2026-07-01)

Three legality-verified 60-card seed genomes built (`scripts/build_coevo_seeds.py`
→ `agent/decks/coevo_seeds/{grimmsnarl,dragapult,clefairy}.csv`). Each = real
archetype Pokémon core + in-pool trainer shell (no Iono/Arven/Prof Research →
Dawn/Hilda/Cheren/Judge/Ultra Ball/Boss's Orders/Buddy-Buddy Poffin/Pokégear).
**Verified: 60/60 cards in-pool (`scripts/check_deck_legal.py`), ≤4-copy rule, and
the ENGINE accepts each at `battle_start` + plays full games to a prize result
(`scripts/validate_deck.py`, 3 games each).**

- **grimmsnarl** (16 Pokémon / 9 {D} / 35 trainers): 4 Marnie's Impidimp, 2 Morgrem,
  3 Marnie's Grimmsnarl ex, 3 Munkidori, 2 Snorunt, 2 Froslass. Plan: Froslass +
  Munkidori spread counters, Grimmsnarl ex Shadow Bullet finisher; Punk Up self-accels.
- **dragapult** (15 Pokémon / 7 {P} + 4 {R} / 34 trainers): 4 Dreepy, 2 Drakloak,
  3 Dragapult ex, 2 Duskull, 1 Dusclops, 2 Dusknoir, 1 Budew. Two-type (Dragon costs
  {R}{P}); Phantom Dive spread + Dusknoir Cursed Blast; Budew item-lock.
- **clefairy** (13 Pokémon / 8 {P} / 39 trainers): 4 Clefairy, 3 Clefable, 4 Dunsparce,
  2 Dudunsparce. Single-prize Clefable-Metronome toolbox + Clefairy Follow-Me gust +
  Dudunsparce draw. **Best-effort reconstruction — the NAIC "Clefairy" list was not
  verified; our pool's Clefairy is a Follow-Me disruptor, not a big attacker. Treat as
  the weakest of the three; the coevolution search should be allowed to reshape it.**

These are *starting genomes*, not tuned lists — the whole point of the coevolution
search is to discover the optimal in-pool 60. Next: register them (+ Archaludon,
Alakazam) as the initial population and start warm-start-fine-tune fitness evaluation.
