# Chinese Incremental Learning System

A system for organizing Chinese learning materials using
i+1 sentence selection across multiple videos.

## Overview

The system processes YouTube videos with Chinese subtitles to create
an optimal learning sequence. It selects sentences that introduce
exactly one new word based on what you already know, then
incrementally builds your vocabulary across multiple videos.

## Quick Start

```bash
# Process multiple videos in sequence
python3 process_videos.py video_urls.txt
```

## How It Works

`process_videos.py` is the main entry point. For each video it:

1. Downloads subtitles and audio (`prepare_vtt_data.py`)
2. Segments sentences into words with pkuseg (`enhance_csv.py`)
3. Runs i+1 selection to pick sentences with exactly one new
   word (`selection.py`, `organizer.py`)
4. Generates audio clips and TTS for target words, adds pinyin
   and cedict definitions (`generate_audio.py`)
5. Appends results to `data_files/all_sentences.csv`
6. Adds new words to `known` so the next video builds on them

Target words are filtered to only include words that appear in
CC-CEDICT (`words/cedict_ts.u8`). Non-cedict segments from the
segmenter are re-split via greedy longest-match against the
dictionary.

## Anki Integration

Export your Anki deck as "Selected Notes.txt" and run:

```bash
python3 anki_to_known.py
```

This extracts the Simplified column into `known`, so the
selection algorithm skips words you already know.

## Directory Structure

- `data_files/` - CSVs, audio files, and VTT subtitles
- `audio_segments/` - Audio clips for sentences and word TTS
- `words/cedict_ts.u8` - CC-CEDICT dictionary
- `words/100k` - Word frequency data (TSV: `Vocab`, `Count`)
- `known` - Known words (one per line), updated after each video

## Configuration

- **Known Words**: `known` file with pre-existing vocabulary
  (generate from Anki with `anki_to_known.py`)
- **Video URLs**: Text file with one YouTube URL per line
  (lines starting with `#` are ignored)

## Output

- `data_files/all_sentences.csv` - Cumulative learning sequence
- Columns include: `Sentence`, `New_Words`, `Word_Rank`,
  `word_definition`, `audio`, `word_audio`, `sentence_pinyin`,
  `word_pinyin`, `video_url`
