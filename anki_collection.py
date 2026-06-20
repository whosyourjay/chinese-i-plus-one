"""Shared read-only helpers for extracting Chinese vocabulary notes from Anki."""

from __future__ import annotations

import html
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ANKI_FIELD_SEP = "\x1f"
DEFAULT_PROFILE = "Pepe"
DEFAULT_DECKS = (
    "First > Second > Chinese > Words > Words 1k",
    "First > Second > Chinese > Words > Words 1k-3k",
    "First > Second > Chinese > Words > Spoonfed 3k-6.5k",
    "First > Second > Chinese > Words > Youtube 6k-11k",
)
DEFAULT_SENTENCE_FIELDS = (
    "SentenceSimplified",
    "Sentence",
    "example sentence",
)

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
SOUND_RE = re.compile(r"\[sound:([^\]]+)\]")


@dataclass
class NoteRow:
    note_id: int
    note_type_id: int
    note_type: str
    decks: tuple[str, ...]
    field_values: dict[str, str]
    sentence: str
    normalized_sentence: str
    card_count: int
    known_card_count: int
    total_reps: int
    max_interval: int
    total_lapses: int
    first_due: int

    @property
    def key_num(self) -> int:
        key = self.field_values.get("Key", "")
        try:
            return int(key)
        except ValueError:
            return -1

    @property
    def word(self) -> str:
        return self.field_values.get("Simplified", "")

    @property
    def key_sort(self) -> tuple[int, str, int]:
        key_num = self.key_num if self.key_num >= 0 else 10**12
        return key_num, self.word, self.note_id

    @property
    def latest_sort(self) -> tuple[int, int]:
        return self.key_num, self.note_id


def anki_profile_collection(profile: str) -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "Anki2"
        / profile
        / "collection.anki2"
    )


DEFAULT_COLLECTION = anki_profile_collection(DEFAULT_PROFILE)


def display_deck_name(db_name: str) -> str:
    return db_name.replace(ANKI_FIELD_SEP, " > ")


def db_deck_name(display_name: str) -> str:
    return display_name.replace(" > ", ANKI_FIELD_SEP)


def normalize_sentence(value: str) -> str:
    value = html.unescape(value or "")
    value = SOUND_RE.sub("", value)
    value = TAG_RE.sub("", value)
    value = unicodedata.normalize("NFKC", value)
    value = SPACE_RE.sub("", value)
    return value.strip()


def strip_for_report(value: str) -> str:
    value = html.unescape(value or "")
    value = TAG_RE.sub("", value)
    value = SPACE_RE.sub(" ", value)
    return value.strip()


def truncate(value: str, limit: int = 100) -> str:
    value = strip_for_report(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def first_field(field_values: dict[str, str], *names: str) -> str:
    for name in names:
        value = field_values.get(name, "")
        if value:
            return value
    return ""


def connect_read_only(collection: Path) -> sqlite3.Connection:
    uri_path = collection.as_posix().replace(" ", "%20")
    con = sqlite3.connect(f"file:{uri_path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    con.create_collation(
        "unicase",
        lambda a, b: (a.casefold() > b.casefold()) - (a.casefold() < b.casefold()),
    )
    return con


def load_fields(con: sqlite3.Connection) -> dict[int, list[str]]:
    rows = con.execute("select ntid, ord, name from fields order by ntid, ord").fetchall()
    fields: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        fields[row["ntid"]].append(row["name"])
    return dict(fields)


def load_note_types(con: sqlite3.Connection) -> dict[int, str]:
    rows = con.execute("select id, name from notetypes").fetchall()
    return {row["id"]: row["name"] for row in rows}


def resolve_sentence_field(
    field_names: Iterable[str],
    preferred_fields: Iterable[str],
) -> str | None:
    names = list(field_names)
    by_casefold = {name.casefold(): name for name in names}
    for preferred in preferred_fields:
        match = by_casefold.get(preferred.casefold())
        if match:
            return match
    for name in names:
        lowered = name.casefold()
        if "sentence" in lowered and "audio" not in lowered:
            return name
    return None


def load_target_notes(
    con: sqlite3.Connection,
    deck_names: tuple[str, ...],
    preferred_sentence_fields: tuple[str, ...] = DEFAULT_SENTENCE_FIELDS,
    include_new: bool = False,
) -> tuple[list[NoteRow], list[str]]:
    fields_by_type = load_fields(con)
    note_types = load_note_types(con)
    placeholders = ",".join("?" for _ in deck_names)
    new_filter = "" if include_new else "and c.type != 0"
    rows = con.execute(
        f"""
        select
          n.id as note_id,
          n.mid as note_type_id,
          n.flds as flds,
          d.name as deck_name,
          c.id as card_id,
          c.type as type,
          c.queue as queue,
          c.reps as reps,
          c.ivl as interval,
          c.lapses as lapses,
          c.due as due
        from cards c
        join notes n on n.id = c.nid
        join decks d on d.id = c.did
        where d.name in ({placeholders})
        {new_filter}
        """,
        deck_names,
    ).fetchall()

    grouped: dict[int, dict[str, object]] = {}
    warnings: list[str] = []
    for row in rows:
        note_id = row["note_id"]
        if note_id not in grouped:
            grouped[note_id] = {
                "note_type_id": row["note_type_id"],
                "flds": row["flds"],
                "decks": set(),
                "card_count": 0,
                "known_card_count": 0,
                "total_reps": 0,
                "max_interval": 0,
                "total_lapses": 0,
                "first_due": row["due"],
            }
        item = grouped[note_id]
        item["decks"].add(display_deck_name(row["deck_name"]))  # type: ignore[union-attr]
        item["card_count"] = int(item["card_count"]) + 1
        if int(row["type"]) != 0:
            item["known_card_count"] = int(item["known_card_count"]) + 1
        item["total_reps"] = int(item["total_reps"]) + int(row["reps"])
        item["max_interval"] = max(int(item["max_interval"]), int(row["interval"]))
        item["total_lapses"] = int(item["total_lapses"]) + int(row["lapses"])
        item["first_due"] = min(int(item["first_due"]), int(row["due"]))

    notes: list[NoteRow] = []
    warned_types: set[int] = set()
    for note_id, item in grouped.items():
        note_type_id = int(item["note_type_id"])
        field_names = fields_by_type.get(note_type_id, [])
        sentence_field = resolve_sentence_field(field_names, preferred_sentence_fields)
        if sentence_field is None:
            if note_type_id not in warned_types:
                warnings.append(
                    f"Skipping note type {note_type_id} "
                    f"({note_types.get(note_type_id, 'unknown')}): no sentence field found"
                )
                warned_types.add(note_type_id)
            continue

        raw_values = str(item["flds"]).split(ANKI_FIELD_SEP)
        field_values = {
            name: raw_values[index] if index < len(raw_values) else ""
            for index, name in enumerate(field_names)
        }
        sentence = field_values.get(sentence_field, "")
        normalized = normalize_sentence(sentence)
        if not normalized:
            continue

        notes.append(
            NoteRow(
                note_id=note_id,
                note_type_id=note_type_id,
                note_type=note_types.get(note_type_id, str(note_type_id)),
                decks=tuple(sorted(item["decks"])),  # type: ignore[arg-type]
                field_values=field_values,
                sentence=sentence,
                normalized_sentence=normalized,
                card_count=int(item["card_count"]),
                known_card_count=int(item["known_card_count"]),
                total_reps=int(item["total_reps"]),
                max_interval=int(item["max_interval"]),
                total_lapses=int(item["total_lapses"]),
                first_due=int(item["first_due"]),
            )
        )

    return notes, warnings


def duplicate_groups(notes: Iterable[NoteRow], keep_latest: bool = True) -> list[list[NoteRow]]:
    grouped: dict[str, list[NoteRow]] = defaultdict(list)
    for note in notes:
        grouped[note.normalized_sentence].append(note)

    groups: list[list[NoteRow]] = []
    for group in grouped.values():
        if len(group) <= 1:
            continue
        if keep_latest:
            groups.append(sorted(group, key=lambda note: note.latest_sort, reverse=True))
        else:
            groups.append(sorted(group, key=lambda note: note.key_sort))
    return sorted(groups, key=lambda group: group[0].key_sort)


def notes_from_collection(
    collection: Path = DEFAULT_COLLECTION,
    display_decks: tuple[str, ...] = DEFAULT_DECKS,
    preferred_sentence_fields: tuple[str, ...] = DEFAULT_SENTENCE_FIELDS,
    include_new: bool = False,
) -> tuple[list[NoteRow], list[str]]:
    deck_names = tuple(db_deck_name(name) for name in display_decks)
    con = connect_read_only(collection)
    try:
        return load_target_notes(con, deck_names, preferred_sentence_fields, include_new)
    finally:
        con.close()
