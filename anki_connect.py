"""Thin wrapper around AnkiConnect (https://ankiweb.net/shared/info/2055492159).

Anki must be running with the add-on installed. All write paths go through this
module so that note/card edits are tracked by Anki itself - including bumping
mod/usn and registering media files in collection.media.db2.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


ANKICONNECT_URL = "http://127.0.0.1:8765"
ANKICONNECT_VERSION = 6


class AnkiConnectError(RuntimeError):
    pass


def request(action: str, **params: Any) -> Any:
    payload = json.dumps(
        {"action": action, "version": ANKICONNECT_VERSION, "params": params}
    ).encode("utf-8")
    req = urllib.request.Request(
        ANKICONNECT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise AnkiConnectError(
            f"Could not reach AnkiConnect at {ANKICONNECT_URL}. Is Anki running?"
        ) from exc
    if body.get("error"):
        raise AnkiConnectError(f"{action} failed: {body['error']}")
    return body.get("result")


def notes_info(note_ids: list[int]) -> list[dict[str, Any]]:
    """Return one entry per existing note. Unknown ids are dropped silently;
    AnkiConnect returns {} placeholders for them and we filter those out."""
    if not note_ids:
        return []
    raw = request("notesInfo", notes=note_ids)
    return [note for note in raw if note.get("noteId")]


def update_note_fields(note_id: int, fields: dict[str, str]) -> None:
    request("updateNoteFields", note={"id": note_id, "fields": fields})


def add_tags(note_ids: list[int], tags: str) -> None:
    if not note_ids:
        return
    request("addTags", notes=note_ids, tags=tags)


def store_media_file(filename: str, path: str, overwrite: bool = False) -> str:
    return request(
        "storeMediaFile",
        filename=filename,
        path=path,
        deleteExisting=overwrite,
    )


def field_order(note: dict[str, Any]) -> list[str]:
    return [
        name
        for name, _ in sorted(
            note["fields"].items(), key=lambda kv: kv[1]["order"]
        )
    ]


def field_values(note: dict[str, Any]) -> dict[str, str]:
    return {name: data["value"] for name, data in note["fields"].items()}


def resolve_field(field_names: list[str], *candidates: str) -> str:
    by_casefold = {name.casefold(): name for name in field_names}
    for candidate in candidates:
        actual = by_casefold.get(candidate.casefold())
        if actual is not None:
            return actual
    raise ValueError(f"None of these fields exist: {', '.join(candidates)}")


def optional_resolve_field(field_names: list[str], *candidates: str) -> str | None:
    by_casefold = {name.casefold(): name for name in field_names}
    for candidate in candidates:
        actual = by_casefold.get(candidate.casefold())
        if actual is not None:
            return actual
    return None
