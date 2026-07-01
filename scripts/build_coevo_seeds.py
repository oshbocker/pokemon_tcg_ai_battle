"""Build legality-verified 60-card coevolution SEED decklists.

Each seed = a real competitive archetype's Pokémon core (verified in-pool) + a
trainer/energy shell rebuilt from OUR curated pool (no Iono/Arven/Prof Research/
Nest Ball — see rl_research/META_RESEARCH_2026-07-01.md). Emits one CSV of 60 Card
IDs per deck to agent/decks/coevo_seeds/, asserting sum==60 and the 4-copy rule
(basic energy exempt). Card names are validated against data/EN_Card_Data.csv.

    uv run python scripts/build_coevo_seeds.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
with open(REPO / "data" / "EN_Card_Data.csv", newline="") as _f:
    CARDS = {r["Card ID"]: r for r in csv.DictReader(_f)}
STAGE = "Stage (Pokémon)/Type (Energy and Trainer)"
BASIC_ENERGY_IDS = {"2", "5", "7"}  # Basic {R}/{P}/{D} Energy — exempt from the 4-copy cap

# id -> count.  (comments = card name for readability; source of truth is the DB)
GRIMMSNARL = {  # Dark damage-spread + disruption control
    "646": 4,  # Marnie's Impidimp   (Filch draw / 70HP basic)
    "647": 2,  # Marnie's Morgrem
    "648": 3,  # Marnie's Grimmsnarl ex  (Punk Up: fetch 5 {D}; Shadow Bullet 180+30)
    "112": 3,  # Munkidori  (Adrena-Brain: move 3 counters w/ {D} attached)
    "103": 2,  # Snorunt (TWM)
    "104": 2,  # Froslass (TWM)  (Freezing Shroud: ping every Ability-mon each Checkup)
    "7": 9,  # Basic {D} Energy
    "1121": 4,  # Ultra Ball
    "1086": 4,  # Buddy-Buddy Poffin  (Impidimp/Snorunt are <=70HP)
    "1079": 3,  # Rare Candy  (Impidimp -> Grimmsnarl ex)
    "1231": 4,  # Dawn  (search a full evo line)
    "1225": 2,  # Hilda  (evo + energy)
    "1224": 3,  # Cheren
    "1213": 2,  # Judge  (disruption)
    "1182": 3,  # Boss's Orders  (gust)
    "1152": 3,  # Poké Pad  (grab non-ex pieces)
    "1097": 3,  # Night Stretcher
    "1122": 2,  # Pokégear 3.0
    "1227": 2,  # Lillie's Determination
}

DRAGAPULT = {  # Dragon tempo + bench-spread
    "119": 4,  # Dreepy
    "120": 2,  # Drakloak  (Recon Directive draw)
    "121": 3,  # Dragapult ex  (Phantom Dive {R}{P} 200 + 6 bench counters)
    "131": 2,  # Duskull
    "132": 1,  # Dusclops
    "133": 2,  # Dusknoir  (Cursed Blast: 13 counters, self-KO)
    "235": 1,  # Budew  (Itchy Pollen item-lock)
    "5": 7,  # Basic {P} Energy
    "2": 4,  # Basic {R} Energy
    "1121": 4,  # Ultra Ball
    "1086": 4,  # Buddy-Buddy Poffin  (Dreepy/Duskull/Budew <=70HP)
    "1079": 4,  # Rare Candy  (Dreepy->Dragapult ex, Duskull->Dusknoir)
    "1231": 4,  # Dawn
    "1225": 2,  # Hilda
    "1224": 3,  # Cheren
    "1182": 3,  # Boss's Orders
    "1152": 3,  # Poké Pad
    "1097": 3,  # Night Stretcher
    "1122": 2,  # Pokégear 3.0
    "1227": 2,  # Lillie's Determination
}

CLEFAIRY = {  # single-prize Metronome toolbox (low-variance; best-effort — NAIC list unverified)
    "1039": 4,  # Clefairy (POR)  (Follow Me: gust opp bench; Flop 30)
    "958": 3,  # Clefable (ASC)  (Metronome: copy opp attack for CC; Magical Shot 100)
    "65": 4,  # Dunsparce (TEF)  (draw engine basic)
    "66": 2,  # Dudunsparce (TEF)  (Run Away Draw: draw 3)
    "5": 8,  # Basic {P} Energy
    "1121": 4,  # Ultra Ball
    "1086": 4,  # Buddy-Buddy Poffin  (Clefairy/Dunsparce <=70HP)
    "1152": 4,  # Poké Pad
    "1224": 4,  # Cheren
    "1213": 4,  # Judge
    "1182": 3,  # Boss's Orders  (double-gust w/ Clefairy Follow Me)
    "1097": 3,  # Night Stretcher
    "1122": 3,  # Pokégear 3.0
    "1199": 3,  # Lacey
    "1227": 3,  # Lillie's Determination
    "1192": 2,  # Carmine
    "1225": 2,  # Hilda
}

DECKS = {"grimmsnarl": GRIMMSNARL, "dragapult": DRAGAPULT, "clefairy": CLEFAIRY}


def validate(name: str, deck: dict[str, int]) -> list[str]:
    total = sum(deck.values())
    errs = []
    if total != 60:
        errs.append(f"{name}: total {total} != 60")
    for cid, n in deck.items():
        if cid not in CARDS:
            errs.append(f"{name}: unknown card id {cid}")
            continue
        if n > 4 and cid not in BASIC_ENERGY_IDS:
            errs.append(f"{name}: {CARDS[cid]['Card Name']} x{n} exceeds 4-copy limit")
    if errs:
        raise SystemExit("DECK VALIDATION FAILED:\n  " + "\n  ".join(errs))
    return [cid for cid, n in deck.items() for _ in range(n)]  # expand to 60 ids


def main() -> None:
    out_dir = REPO / "agent" / "decks" / "coevo_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, deck in DECKS.items():
        ids = validate(name, deck)
        (out_dir / f"{name}.csv").write_text("\n".join(ids) + "\n")
        npok = sum(n for c, n in deck.items() if "Pokémon" in CARDS[c][STAGE])
        nen = sum(n for c, n in deck.items() if "Energy" in CARDS[c][STAGE])
        print(
            f"{name:12} 60 cards  ({npok} Pokémon / {nen} energy / {60 - npok - nen} trainers)"
            f"  -> {out_dir.relative_to(REPO)}/{name}.csv"
        )


if __name__ == "__main__":
    main()
