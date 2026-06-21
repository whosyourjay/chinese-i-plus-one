# Chinese Incremental Learning System

A system for organizing Chinese learning materials using
i+1 sentence selection across multiple videos.

The system processes YouTube videos with Chinese subtitles to create
an optimal learning sequence. It selects sentences that introduce
exactly one new word based on what you already know, then
incrementally builds your vocabulary across multiple videos.

## Quick Start

```bash
# Install Python deps (see Installation below for optional extras)
pip install -r requirements.txt

# Process multiple videos in sequence
python3 video_pipeline/process_videos.py video_urls.txt
```

## Installation

System tools (install via your OS package manager): `yt-dlp`, `ffmpeg`.

Python deps live in `requirements.txt`. The file has two commented-out
optional sections:

- **Audio transcription** (`funasr`, `torchaudio`, `numpy<2`) — adds an
  `audio_transcription` + `audio_language` column to each card by running
  SenseVoice over the clipped sentence audio. First run downloads a ~1GB
  model under `~/.cache/modelscope/`; per-clip latency is ~200 ms on CPU.
  When these packages aren't installed the pipeline simply skips the step
  and leaves the columns blank.

Uncomment the lines under each section in `requirements.txt` to opt in,
then re-run `pip install -r requirements.txt`.

## How It Works

`process_videos.py` is the main entry point. For each video it:

1. Downloads subtitles and audio (`prepare_vtt_data.py`)
2. Segments sentences into words with pkuseg (`enhance_csv.py`)
3. Runs i+1 selection to pick sentences with exactly one new
   word (`selection.py`, `organizer.py`)
4. Generates audio clips and TTS for target words, adds pinyin
   and cedict definitions (`generate_audio.py`)
5. Transcribes each sentence audio clip with SenseVoice and
   records the detected language (`transcribe_audio.py`, optional —
   see Installation)
6. Appends results to `data_files/all_sentences.csv`
7. Adds new words to `known` so the next video builds on them

## Anki Integration

Export your Anki deck as "Selected Notes.txt" and run:

```bash
python3 anki_to_known.py
```

This extracts the Simplified column into `known`, so the
selection algorithm skips words you already know.

## Configuration

- **Known Words**: `known` file with pre-existing vocabulary
  (generate from Anki with `anki_to_known.py`)
- **Video URLs**: Text file with one YouTube URL per line
  (lines starting with `#` are ignored)

## Output

- `data_files/all_sentences.csv` - Cumulative learning sequence.

## Directory Structure

- `data_files/` - CSVs, audio files, and VTT subtitles
- `audio_segments/` - Audio clips for sentences and word TTS
- `words/cedict_ts.u8` - CC-CEDICT dictionary
- `words/100k` - Word frequency data (TSV: `Vocab`, `Count`)
- `known` - Known words (one per line), updated after each video
