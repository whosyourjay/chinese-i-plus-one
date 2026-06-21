#!/usr/bin/env python3
"""Transcribe sentence audio clips with SenseVoice (FunASR).

Loaded once per process; reused across all clips for the video. The model is
non-autoregressive so per-clip latency is on the order of tens of ms on CPU
for short Chinese clips.
"""

from __future__ import annotations

import re
from pathlib import Path

_MODEL = None
_TAG_RE = re.compile(r"<\|[^|]*\|>")
_LANG_RE = re.compile(r"<\|([a-z]{2})\|>")


def is_available() -> bool:
    """True if SenseVoice transcription can run (funasr + torchaudio present)."""
    try:
        import funasr  # noqa: F401
        import torchaudio  # noqa: F401
        return True
    except ImportError:
        return False


def get_model():
    """Lazily load SenseVoice-small. First call downloads ~470MB."""
    global _MODEL
    if _MODEL is None:
        from funasr import AutoModel
        _MODEL = AutoModel(
            model="iic/SenseVoiceSmall",
            disable_update=True,
            log_level="ERROR",
        )
    return _MODEL


def parse(raw: str) -> tuple[str, str]:
    """Split SenseVoice's raw output `<|zh|><|NEUTRAL|>...<|woitn|>text` into
    (language, clean_text). Language is the first 2-letter tag (zh/en/ja/ko/yue)
    or '' if absent."""
    m = _LANG_RE.search(raw)
    lang = m.group(1) if m else ""
    return lang, _TAG_RE.sub("", raw).strip()


def _generate(inputs, **kw):
    return get_model().generate(
        input=inputs,
        language="auto",
        use_itn=True,
        disable_pbar=True,
        **kw,
    )


def transcribe(audio_path: str | Path) -> tuple[str, str]:
    """Transcribe one audio file. Returns (language, text)."""
    res = _generate(str(audio_path))
    return parse(res[0].get("text", "")) if res else ("", "")


def transcribe_many(paths: list[Path], batch_size: int = 8) -> list[tuple[str, str]]:
    """Batch-transcribe a list of audio paths. Returns one (language, text)
    tuple per path, in the same order."""
    out: list[tuple[str, str]] = [("", "")] * len(paths)
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        try:
            res = _generate([str(p) for p in chunk], batch_size=len(chunk))
        except Exception as exc:
            print(f"  transcribe batch failed at {start}: {exc}")
            continue
        for i, r in enumerate(res):
            out[start + i] = parse(r.get("text", ""))
        print(f"  transcribed {min(start + batch_size, len(paths))}/{len(paths)}")
    return out


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        lang, text = transcribe(arg)
        print(f"{arg}\t{lang}\t{text}")
