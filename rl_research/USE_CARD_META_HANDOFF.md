# Handoff prompt — implement `use_card_meta` + run the novel-card kill-criterion

Paste the block below into a fresh Claude Code session in this repo. It is
self-contained but assumes the session can read `rl_research/STRATEGY_WRITEUP_LOG.md`,
`rl_research/COEVOLUTIONARY_DECK_SEARCH.md`, and the memory index.

---

## PROMPT

We are building a coevolutionary deck-search where a trained agent is the fitness
function for a deck (design: `rl_research/COEVOLUTIONARY_DECK_SEARCH.md`; status +
results: the three `2026-07-01` entries in `rl_research/STRATEGY_WRITEUP_LOG.md`).

**What's already established (don't re-litigate):** The §4.1 kill-criterion PASSED —
warm-start fine-tune (`--init-ckpt`) recovers parent strength cheaply where cold
training collapses. A high-n eval then showed the sharper result: on a **1-card swap**
(`agent/decks/archaludon_judge_swap.csv`), zero-shot (parent on the swapped deck, no
fine-tune) is already ~ref on aggregate, but fine-tuning unlocks a **+11.9pp
Alakazam** gain that zero-shot misses — because the swapped card (Judge) has *latent*
value the policy only realizes after fine-tuning. **Conclusion adopted:** two-stage
fitness = zero-shot coarse pre-filter → warm-start fine-tune on survivors.

**The open bottleneck this task resolves.** Everything above is the *easy* case: a
1-card swap barely perturbs the card-ID embedding, whose rows are already trained.
The real search needs **novel-card** mutations — cards the parent essentially never
saw, whose embedding rows are cold. There, both zero-shot and warm-start are expected
to degrade badly. The design's proposed fix is a **frozen static card-metadata
feature** (`use_card_meta` ablation flag): give every card a sensible representation
from its printed attributes (type/HP/retreat/weakness/stage/ex-mega-ACE flags) so a
novel card isn't a blank embedding row. Docs deliberately left this out
(`docs/rl-obs-action.md` §0: "Static `card_table` metadata … deliberately not baked
in; the ID embedding is expected to learn it from outcomes").

### Your task, in two parts

**PART A — implement `use_card_meta`, mirroring the existing `use_option_rank`
ablation.** Trace `use_option_rank` first; it is the exact template (grep it: it
touches `src/ptcg_battle/model.py` `ModelConfig` + forward, `scripts/train_selfplay.py`
flag/print, `tests/test_model.py`, `docs/rl-obs-action.md`).

1. **Frozen metadata table.** Add a torch-free module `src/ptcg_battle/card_meta.py`
   that builds a `[CARD_VOCAB, F]` `float32` numpy table keyed by Card ID (row N =
   card N, row 0 = PAD/zeros), from `data/EN_Card_Data.csv`. `CARD_VOCAB=1268`, ids
   are 1-based and dense (see `src/ptcg_battle/encoding.py:33-43`; the CSV `Card ID`
   column aligns with the engine's `all_card_data()` ids). Feature columns (document
   the exact layout): type one-hot (parse `Type` `{M}/{P}/…`, ~12 incl. Dragon/none),
   HP normalized scalar (`HP/340`) + a small HP bucket one-hot, retreat cost scalar,
   weakness type one-hot, stage one-hot (Basic/Stage1/Stage2/none from the
   `Stage (Pokémon)/Type…` column), and boolean flags from `Rule` (`Pokémon ex`,
   `Mega Pokémon ex`, `ACE SPEC`) + is-Pokémon / is-Trainer / is-Energy. Keep it
   deterministic and cached (build once at import).
2. **Model wiring** (`src/ptcg_battle/model.py`): add `use_card_meta: bool = False`
   to `ModelConfig` (line ~61; default OFF = opt-in ablation). In `PtcgNet.__init__`,
   `register_buffer` the frozen table and add a learned `nn.Linear(F, d)` projection
   (`card_meta_proj`). In `forward`, when `cfg.use_card_meta`, add
   `card_meta_proj(META[batch["ent_card"]])` to the entity card embedding and the
   same for `batch["opt_card"]` on options — mirroring the `+ self.rank_emb(...)` add
   at `model.py:224-225`. The table is frozen (buffer, no grad); only the projection
   learns.
3. **Flag wiring** (`scripts/train_selfplay.py`): add `--use-card-meta` (store_true)
   and set it into the cfg exactly where `--no-option-rank` is handled (~line 353),
   and in the warm-start branch (~line 341, `--init-ckpt`). Add it to the startup
   print line. `scripts/eval.py` already rebuilds the net from the checkpoint's cfg,
   so eval needs no change (verify).
4. **Tests + gate.** Mirror `tests/test_model.py:94` (toggle changes outputs) and add
   a `tests/test_card_meta.py` asserting the table shape `[CARD_VOCAB, F]`, PAD row is
   zero, and a few spot-checked cards (e.g. Archaludon ex `130`→Metal/ex flag, Basic
   {D} Energy `7`→energy flag). Keep the table build torch-free. Run
   `uv run python scripts/prepare.py --check` green before proceeding.

**PART B — the novel-card kill-criterion (the decisive test).** Does `use_card_meta`
let warm-start recover strength on a mutation that adds a card the parent NEVER saw,
where the metadata-OFF baseline craters?

1. **Build a novel-card swap deck.** Start from `agent/kaggle_agents/archaludon_deck.csv`
   and swap 1–2 cards for a card that is (a) NOT in the Archaludon parent's training
   deck(s), (b) deck-coherent (still a legal Metal deck), (c) ideally matchup-relevant.
   Pick a card whose embedding row is genuinely cold — verify via the parent's training
   data / that it never appeared. Write it to `agent/decks/archaludon_novel_swap.csv`
   and validate with `scripts/check_deck_legal.py` (60/60 in-pool) AND
   `scripts/validate_deck.py` (engine accepts + plays).
2. **The gotcha you must handle:** the parent (`outputs/probe_archaludon_medium_long/
   best.pt`) was trained with `use_card_meta=OFF`, so it has no `card_meta_proj`
   weights. `--init-ckpt` loads `strict=False` (already), so with `--use-card-meta` the
   projection initializes fresh and must fine-tune. A *clean* test of the feature's
   warm-start benefit ideally uses a **card_meta-ON parent**. Decide and document one:
   (a) quick path — run the kill-criterion from the OFF parent and see if a fresh proj
   still helps within the 30-iter budget; (b) clean path — first train a short
   card_meta-ON parent (Colab/L4; local is CPU-only, see below), then warm-start from
   it. Recommend (b) for the definitive number, (a) as a fast signal first.
3. **Run the 2×2 (mirror the coevo_kill recipe).** Reuse `scripts/train_selfplay.py`
   and the exact eval command block from the `2026-07-01` "kill-criterion" work (see
   `rl_research/coevo_kill_eval.log` + `scripts/run_coevo_kill_eval.sh`): for
   {`use_card_meta` OFF, ON} run warm-start fine-tune (30 iters, `--init-ckpt`, LR
   3e-5→1e-5, `--opp-manifest agent/opponents/mixed_pool.json`, `--gate --gate-vs pool`)
   on `archaludon_novel_swap.csv`, then high-n side-swapped `scripts/eval.py`
   (160 games × the 7 mixed_pool opponents, Wilson CIs) for: ref (parent/orig),
   zero-shot (parent/novel-swap), warm-OFF, warm-ON. Local is **CPU-only** (no GPU) —
   30-iter warm runs took ~2h; keep budgets small or run on Colab
   (`notebooks/colab_selfplay_archaludon.ipynb` is the L4 harness).
4. **Success criterion.** `use_card_meta` PASSES if, on the novel-card swap, warm-ON
   recovers toward ref (like the 1-card swap did, ~within a few pp) while warm-OFF
   underperforms (cold embedding row) — i.e., metadata rescues novel-card warm-start.
   If ON ≈ OFF, the feature does NOT unlock novel-card mutations → report that
   honestly; the search stays bounded to near-neighbor swaps and the next lever is
   different (e.g. longer fine-tunes, or embedding-row init from a similar card).

### Constraints / definition of done
- Always run Python via `uv`; `uv run python scripts/prepare.py --check` must be green
  (ruff+pyright+pytest) before commits. Branch off `main` first; commit only when asked.
- Record results + verdict in `rl_research/STRATEGY_WRITEUP_LOG.md` (dated entry) and
  update the `coevolutionary-deck-search` memory. Per the `unproven-work-stays-out-of-
  writeup` memory, this graduates into the writeup only if the full search later shows
  a real deck improvement — keep it a dated experiment log until then.
- Deliverables: `card_meta.py` + model/flag/test wiring (prepare green); the
  `archaludon_novel_swap.csv` deck; the 4-arm eval CSVs under `outputs/coevo_kill/` (or
  a new `outputs/card_meta_kill/`); and a strategy-log entry with the table + the
  PASS/FAIL verdict on whether metadata rescues novel-card warm-start.

### Key anchors
- `use_option_rank` template: `src/ptcg_battle/model.py:61,224` · `scripts/train_selfplay.py:353` ·
  `tests/test_model.py:94` · `docs/rl-obs-action.md:136`.
- Card-ID vocab / provenance: `src/ptcg_battle/encoding.py:33-43` (`CARD_VOCAB=1268`, ids 1..1267).
- Metadata source: `data/EN_Card_Data.csv` (cols: `Type`, `HP`, `Retreat`, `Weakness`,
  `Stage (Pokémon)/Type…`, `Rule`).
- Reusable tooling: `scripts/train_selfplay.py` (`--init-ckpt` warm-start), `scripts/eval.py`
  (high-n side-swapped), `scripts/check_deck_legal.py`, `scripts/validate_deck.py`,
  `scripts/run_coevo_kill_eval.sh` (eval recipe), parent ckpt
  `outputs/probe_archaludon_medium_long/best.pt`.
