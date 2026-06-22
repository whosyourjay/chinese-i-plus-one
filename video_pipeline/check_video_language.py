#!/usr/bin/env python3
"""Detect & drop non-Chinese sentences from a video before selection.

Sample N random sentences from sentences_basic.csv and ECAPA-detect each
clip's language. If they are all zh, the whole video is assumed Chinese.
If any are non-zh, run the full detection over every sentence and drop
the non-zh rows from sentences_basic.csv. Running this before selection
ensures non-Chinese sentences never become dependencies of later cards.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect_language import detect_many
from generate_audio import split_audio


SAMPLE_SIZE = 20
ECAPA_BATCH = 32


def _detect_segments(
    df: pd.DataFrame, audio_file: str, tmp_dir: Path
) -> list[str]:
    segments = [
        (row.start_time, row.end_time, row.Sentence)
        for row in df.itertuples()
    ]
    files = split_audio(audio_file, segments, str(tmp_dir), padding=0.2)
    paths = [tmp_dir / f for f in files]
    return detect_many(paths, backend="ecapa", batch_size=ECAPA_BATCH)


def check_and_filter(
    basic_csv: str,
    audio_file: str,
    sample_size: int = SAMPLE_SIZE,
    seed: int = 0,
) -> int:
    """Return number of non-zh sentences dropped from basic_csv."""
    df = pd.read_csv(basic_csv)
    if df.empty:
        return 0
    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=seed)
    with tempfile.TemporaryDirectory(prefix="lang_check_") as tmp:
        tmp_dir = Path(tmp)
        print(f"Sampling {n} sentences for language check...")
        sample_langs = _detect_segments(sample, audio_file, tmp_dir)
        non_zh = sum(1 for lang in sample_langs if lang != "zh")
        if non_zh == 0:
            print(f"All {n} sampled sentences are zh; proceeding.")
            return 0
        print(f"Found {non_zh}/{n} non-zh in sample; running full check...")
        all_langs = _detect_segments(df, audio_file, tmp_dir)

    df["audio_language"] = all_langs
    # Treat detection failures ('') as 'keep' so a transient ECAPA error
    # doesn't drop a valid Chinese clip — clean_bad_video can still revisit.
    keep_mask = df["audio_language"].isin(["zh", ""])
    dropped = df.loc[~keep_mask]
    if dropped.empty:
        print("Full check found no non-zh sentences; proceeding.")
        return 0
    lang_counts = dropped["audio_language"].value_counts().to_dict()
    print(f"Dropping {len(dropped)} non-zh sentences: {lang_counts}")
    df.loc[keep_mask].drop(columns=["audio_language"]).to_csv(basic_csv, index=False)
    return len(dropped)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--basic-csv", default="data_files/sentences_basic.csv")
    p.add_argument("--audio-file", default="data_files/video.mp3")
    p.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    check_and_filter(args.basic_csv, args.audio_file, args.sample_size, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
