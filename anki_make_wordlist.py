#!/usr/bin/env python3
"""Build the known-word list directly from known Anki vocabulary cards.

By default, when multiple notes share the same normalized sentence, the script
keeps the earliest card in deck-block order in the wordlist and omits newer
duplicate-sentence words from the wordlist only. It does not modify Anki.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from anki_collection import (
    DECK_BLOCKS,
    DEFAULT_COLLECTION,
    DEFAULT_SENTENCE_FIELDS,
    NoteRow,
    duplicate_groups,
    first_field,
    notes_from_collection,
    strip_for_report,
)


def word_text(note: NoteRow) -> str:
    return strip_for_report(note.word)


def replacement_map(groups: list[list[NoteRow]]) -> dict[int, NoteRow]:
    replacements: dict[int, NoteRow] = {}
    for group in groups:
        keeper = group[0]
        for note in group[1:]:
            replacements[note.note_id] = keeper
    return replacements


def duplicate_note_ids(groups: list[list[NoteRow]]) -> set[int]:
    return {note.note_id for group in groups for note in group}


def load_segmentation_omissions(
    path: Path,
    grouped_note_ids: set[int],
) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("action") != "replace_sentence":
                continue
            try:
                note_id = int(row.get("note_id", ""))
            except ValueError:
                continue
            if note_id not in grouped_note_ids:
                out[note_id] = row
    return out


def sorted_word_notes(notes: list[NoteRow]) -> list[NoteRow]:
    return sorted(notes, key=lambda note: note.key_sort)


def build_wordlist(
    notes: list[NoteRow],
    segmentation_omissions_path: Path | None = None,
    omit_earlier_duplicate_sentences: bool = True,
) -> tuple[list[str], dict[int, NoteRow], dict[int, dict[str, str]], list[list[NoteRow]]]:
    groups = duplicate_groups(notes)
    omitted_by_note_id = replacement_map(groups) if omit_earlier_duplicate_sentences else {}
    segmentation_omissions = (
        load_segmentation_omissions(
            segmentation_omissions_path,
            duplicate_note_ids(groups),
        )
        if segmentation_omissions_path is not None
        else {}
    )

    words: list[str] = []
    seen: set[str] = set()
    for note in sorted_word_notes(notes):
        if note.note_id in omitted_by_note_id or note.note_id in segmentation_omissions:
            continue
        word = word_text(note)
        if not word or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words, omitted_by_note_id, segmentation_omissions, groups


def build_positioned_wordlist(
    block_notes: list[tuple[str, list[NoteRow]]],
    generated_csv: Path,
    segmentation_omissions_path: Path | None = None,
    omit_earlier_duplicate_sentences: bool = True,
) -> tuple[list[str], list[dict[str, str]], dict[int, NoteRow], dict[int, dict[str, str]], list[list[NoteRow]]]:
    all_notes = [note for _, notes in block_notes for note in notes]
    groups = duplicate_groups(all_notes)
    omitted_by_note_id = replacement_map(groups) if omit_earlier_duplicate_sentences else {}
    segmentation_omissions = (
        load_segmentation_omissions(
            segmentation_omissions_path,
            duplicate_note_ids(groups),
        )
        if segmentation_omissions_path is not None
        else {}
    )

    words: list[str] = []
    positions: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_word(word: str, source: str, source_key: str = "", note_id: str = "") -> None:
        word = strip_for_report(word)
        if not word or word in seen:
            return
        seen.add(word)
        words.append(word)
        positions.append(
            {
                "position": str(len(words)),
                "word": word,
                "source": source,
                "source_key": source_key,
                "note_id": note_id,
            }
        )

    for block_name, notes in block_notes:
        for note in sorted_word_notes(notes):
            if note.note_id in omitted_by_note_id or note.note_id in segmentation_omissions:
                continue
            add_word(
                word_text(note),
                block_name,
                note.field_values.get("Key", ""),
                str(note.note_id),
            )

    if generated_csv.exists():
        with generated_csv.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                add_word(
                    row.get("New_Words", ""),
                    "generated",
                    row.get("Sequence", ""),
                    "",
                )

    return words, positions, omitted_by_note_id, segmentation_omissions, groups


def write_wordlist(words: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for word in words:
            f.write(word)
            f.write("\n")


def write_positions(positions: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("position", "word", "source", "source_key", "note_id")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(positions)


def write_omitted_report(omitted_by_note_id: dict[int, NoteRow], notes: list[NoteRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    notes_by_id = {note.note_id: note for note in notes}
    fieldnames = (
        "omitted_note_id",
        "omitted_key",
        "omitted_word",
        "keeper_note_id",
        "keeper_key",
        "keeper_word",
        "sentence",
        "keeper_pinyin",
        "keeper_translation",
        "keeper_audio",
    )
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for omitted_id in sorted(omitted_by_note_id, key=lambda nid: notes_by_id[nid].key_sort):
            omitted = notes_by_id[omitted_id]
            keeper = omitted_by_note_id[omitted_id]
            keeper_fields = keeper.field_values
            writer.writerow(
                {
                    "omitted_note_id": omitted.note_id,
                    "omitted_key": omitted.field_values.get("Key", ""),
                    "omitted_word": omitted.word,
                    "keeper_note_id": keeper.note_id,
                    "keeper_key": keeper_fields.get("Key", ""),
                    "keeper_word": keeper.word,
                    "sentence": strip_for_report(keeper.sentence),
                    "keeper_pinyin": first_field(
                        keeper_fields,
                        "SentencePinyin.1",
                        "SentencePinyin",
                    ),
                    "keeper_translation": keeper_fields.get("SentenceMeaning", ""),
                    "keeper_audio": keeper_fields.get("SentenceAudio", ""),
                }
            )


def write_segmentation_omitted_report(
    segmentation_omissions: dict[int, dict[str, str]],
    notes: list[NoteRow],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    notes_by_id = {note.note_id: note for note in notes}
    fieldnames = (
        "note_id",
        "key",
        "word",
        "sentence",
        "candidate_tokens",
        "reason",
    )
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for note_id in sorted(segmentation_omissions, key=lambda nid: notes_by_id[nid].key_sort):
            note = notes_by_id[note_id]
            row = segmentation_omissions[note_id]
            writer.writerow(
                {
                    "note_id": note_id,
                    "key": note.field_values.get("Key", ""),
                    "word": word_text(note),
                    "sentence": strip_for_report(note.sentence),
                    "candidate_tokens": row.get("candidate_tokens", ""),
                    "reason": row.get("reason", ""),
                }
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deck",
        action="append",
        dest="decks",
        help="Override deck paths to extract as one block, using ' > ' separators. May be repeated.",
    )
    parser.add_argument(
        "--sentence-field",
        action="append",
        dest="sentence_fields",
        help="Preferred sentence field name. May be repeated.",
    )
    parser.add_argument(
        "--keep-earlier-duplicate-words",
        action="store_true",
        help="Do not omit non-keeper words when duplicate cards share a sentence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("known"),
        help="Known-word output path (default: known).",
    )
    parser.add_argument(
        "--generated-csv",
        type=Path,
        default=Path("data_files/all_sentences.csv"),
        help="Append generated New_Words after Anki deck words (default: data_files/all_sentences.csv).",
    )
    parser.add_argument(
        "--positions-output",
        type=Path,
        default=Path("reports/known_word_positions.tsv"),
        help="TSV path for distinct word positions.",
    )
    parser.add_argument(
        "--omitted-report",
        type=Path,
        default=Path("reports/anki_wordlist_omitted_duplicates.tsv"),
        help="TSV audit report for words omitted from the wordlist.",
    )
    parser.add_argument(
        "--segmentation-actions-report",
        type=Path,
        default=Path("reports/words_1k3k_target_segmentation_actions.tsv"),
        help="TSV action report whose non-duplicate replace_sentence rows are also omitted.",
    )
    parser.add_argument(
        "--segmentation-omitted-report",
        type=Path,
        default=Path("reports/anki_wordlist_omitted_segmentation.tsv"),
        help="TSV audit report for segmentation-based omissions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not DEFAULT_COLLECTION.exists():
        print(f"Collection not found: {DEFAULT_COLLECTION}", file=sys.stderr)
        return 1

    preferred_sentence_fields = tuple(args.sentence_fields or DEFAULT_SENTENCE_FIELDS)
    warnings: list[str] = []

    if args.decks:
        notes, deck_warnings = notes_from_collection(
            display_decks=tuple(args.decks),
            preferred_sentence_fields=preferred_sentence_fields,
        )
        warnings.extend(deck_warnings)
        block_notes = [("anki", notes)]
    else:
        block_notes = []
        for block_name, decks in DECK_BLOCKS:
            notes, deck_warnings = notes_from_collection(
                display_decks=decks,
                preferred_sentence_fields=preferred_sentence_fields,
            )
            warnings.extend(deck_warnings)
            block_notes.append((block_name, notes))

    words, positions, omitted_by_note_id, segmentation_omissions, groups = build_positioned_wordlist(
        block_notes,
        args.generated_csv,
        segmentation_omissions_path=args.segmentation_actions_report,
        omit_earlier_duplicate_sentences=not args.keep_earlier_duplicate_words,
    )
    notes = [note for _, block in block_notes for note in block]

    write_wordlist(words, args.output)
    write_positions(positions, args.positions_output)
    write_omitted_report(omitted_by_note_id, notes, args.omitted_report)
    write_segmentation_omitted_report(
        segmentation_omissions,
        notes,
        args.segmentation_omitted_report,
    )

    print(f"Scanned notes: {len(notes)}")
    print(f"Duplicate sentence groups: {len(groups)}")
    print(f"Omitted non-keeper duplicate-sentence notes: {len(omitted_by_note_id)}")
    print(f"Omitted segmentation replace-sentence notes: {len(segmentation_omissions)}")
    print(f"Known words written: {len(words)}")
    print(f"Wordlist: {args.output}")
    print(f"Positions: {args.positions_output}")
    print(f"Omitted report: {args.omitted_report}")
    print(f"Segmentation omitted report: {args.segmentation_omitted_report}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
