#!/usr/bin/env python3
"""Write `known` from the current Anki word deck (via AnkiConnect).

If `reports/anki_duplicate_sentences.tsv` exists, members of each duplicate
group are dropped except for the survivor (position N-2 — for pairs the keeper,
for triples the middle). Run `python3 repair/duplicate_sentences.py` first to
refresh that TSV.

Legacy: pass a path to read from an Anki text export instead of querying.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import anki_connect

DECK_QUERY = 'deck:"First::Second::Chinese::Words"'
DUPES = Path("reports/anki_duplicate_sentences.tsv")
KNOWN = Path("known")


def load_drop_ids() -> set[int]:
    if not DUPES.exists():
        return set()
    groups: dict[str, list[int]] = defaultdict(list)
    with DUPES.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            groups[r["group_id"]].append(int(r["note_id"]))
    drop: set[int] = set()
    for members in groups.values():
        survivor_pos = len(members) - 2
        for pos, nid in enumerate(members):
            if pos != survivor_pos:
                drop.add(nid)
    return drop


def words_from_anki() -> list[str]:
    drop = load_drop_ids()
    print(f"dropping {len(drop)} bad-group non-survivors")
    note_ids = anki_connect.request("findNotes", query=DECK_QUERY)
    print(f"queried {len(note_ids)} notes in {DECK_QUERY}")
    notes = anki_connect.notes_info(note_ids)
    out: list[str] = []
    for n in notes:
        if n["noteId"] in drop:
            continue
        fv = anki_connect.field_values(n)
        word = (fv.get("Simplified") or fv.get("Word", "")).strip()
        if word:
            out.append(word)
    return out


def words_from_export(path: str) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0].strip().isdigit():
                w = parts[1].strip()
                if w:
                    out.append(w)
    return out


def main() -> int:
    if len(sys.argv) > 1:
        words = words_from_export(sys.argv[1])
    else:
        words = words_from_anki()
    KNOWN.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"wrote {len(words)} words to {KNOWN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
